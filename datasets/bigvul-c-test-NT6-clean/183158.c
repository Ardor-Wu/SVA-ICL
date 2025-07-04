#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

static int ext4_fill_super(struct super_block *sb, void *data, int silent)
{
	char *orig_data = kstrdup(data, GFP_KERNEL);
	struct buffer_head *bh;
	struct ext4_super_block *es = NULL;
	struct ext4_sb_info *sbi;
	ext4_fsblk_t block;
	ext4_fsblk_t sb_block = get_sb_block(&data);
	ext4_fsblk_t logical_sb_block;
	unsigned long offset = 0;
	unsigned long journal_devnum = 0;
	unsigned long def_mount_opts;
	struct inode *root;
	const char *descr;
	int ret = -ENOMEM;
	int blocksize, clustersize;
	unsigned int db_count;
	unsigned int i;
	int needs_recovery, has_huge_files, has_bigalloc;
	__u64 blocks_count;
	int err = 0;
	unsigned int journal_ioprio = DEFAULT_JOURNAL_IOPRIO;
	ext4_group_t first_not_zeroed;

	sbi = kzalloc(sizeof(*sbi), GFP_KERNEL);
	if (!sbi)
		goto out_free_orig;

	sbi->s_blockgroup_lock =
		kzalloc(sizeof(struct blockgroup_lock), GFP_KERNEL);
	if (!sbi->s_blockgroup_lock) {
		kfree(sbi);
		goto out_free_orig;
	}
	sb->s_fs_info = sbi;
	sbi->s_sb = sb;
	sbi->s_inode_readahead_blks = EXT4_DEF_INODE_READAHEAD_BLKS;
	sbi->s_sb_block = sb_block;
	if (sb->s_bdev->bd_part)
		sbi->s_sectors_written_start =
			part_stat_read(sb->s_bdev->bd_part, sectors[1]);

	/* Cleanup superblock name */
	strreplace(sb->s_id, '/', '!');

	/* -EINVAL is default */
	ret = -EINVAL;
	blocksize = sb_min_blocksize(sb, EXT4_MIN_BLOCK_SIZE);
	if (!blocksize) {
		ext4_msg(sb, KERN_ERR, "unable to set blocksize");
		goto out_fail;
	}

	/*
	 * The ext4 superblock will not be buffer aligned for other than 1kB
	 * block sizes.  We need to calculate the offset from buffer start.
	 */
	if (blocksize != EXT4_MIN_BLOCK_SIZE) {
		logical_sb_block = sb_block * EXT4_MIN_BLOCK_SIZE;
		offset = do_div(logical_sb_block, blocksize);
	} else {
		logical_sb_block = sb_block;
	}

	if (!(bh = sb_bread_unmovable(sb, logical_sb_block))) {
		ext4_msg(sb, KERN_ERR, "unable to read superblock");
		goto out_fail;
	}
	/*
	 * Note: s_es must be initialized as soon as possible because
	 *       some ext4 macro-instructions depend on its value
	 */
	es = (struct ext4_super_block *) (bh->b_data + offset);
	sbi->s_es = es;
	sb->s_magic = le16_to_cpu(es->s_magic);
	if (sb->s_magic != EXT4_SUPER_MAGIC)
		goto cantfind_ext4;
	sbi->s_kbytes_written = le64_to_cpu(es->s_kbytes_written);

	/* Warn if metadata_csum and gdt_csum are both set. */
	if (ext4_has_feature_metadata_csum(sb) &&
	    ext4_has_feature_gdt_csum(sb))
		ext4_warning(sb, "metadata_csum and uninit_bg are "
			     "redundant flags; please run fsck.");

	/* Check for a known checksum algorithm */
	if (!ext4_verify_csum_type(sb, es)) {
		ext4_msg(sb, KERN_ERR, "VFS: Found ext4 filesystem with "
			 "unknown checksum algorithm.");
		silent = 1;
		goto cantfind_ext4;
	}

	/* Load the checksum driver */
	if (ext4_has_feature_metadata_csum(sb)) {
		sbi->s_chksum_driver = crypto_alloc_shash("crc32c", 0, 0);
		if (IS_ERR(sbi->s_chksum_driver)) {
			ext4_msg(sb, KERN_ERR, "Cannot load crc32c driver.");
			ret = PTR_ERR(sbi->s_chksum_driver);
			sbi->s_chksum_driver = NULL;
			goto failed_mount;
		}
	}

	/* Check superblock checksum */
	if (!ext4_superblock_csum_verify(sb, es)) {
		ext4_msg(sb, KERN_ERR, "VFS: Found ext4 filesystem with "
			 "invalid superblock checksum.  Run e2fsck?");
		silent = 1;
		ret = -EFSBADCRC;
		goto cantfind_ext4;
	}

	/* Precompute checksum seed for all metadata */
	if (ext4_has_feature_csum_seed(sb))
		sbi->s_csum_seed = le32_to_cpu(es->s_checksum_seed);
	else if (ext4_has_metadata_csum(sb))
		sbi->s_csum_seed = ext4_chksum(sbi, ~0, es->s_uuid,
					       sizeof(es->s_uuid));

	/* Set defaults before we parse the mount options */
	def_mount_opts = le32_to_cpu(es->s_default_mount_opts);
	set_opt(sb, INIT_INODE_TABLE);
	if (def_mount_opts & EXT4_DEFM_DEBUG)
		set_opt(sb, DEBUG);
	if (def_mount_opts & EXT4_DEFM_BSDGROUPS)
		set_opt(sb, GRPID);
	if (def_mount_opts & EXT4_DEFM_UID16)
		set_opt(sb, NO_UID32);
	/* xattr user namespace & acls are now defaulted on */
	set_opt(sb, XATTR_USER);
#ifdef CONFIG_EXT4_FS_POSIX_ACL
	set_opt(sb, POSIX_ACL);
#endif
	/* don't forget to enable journal_csum when metadata_csum is enabled. */
	if (ext4_has_metadata_csum(sb))
		set_opt(sb, JOURNAL_CHECKSUM);

	if ((def_mount_opts & EXT4_DEFM_JMODE) == EXT4_DEFM_JMODE_DATA)
		set_opt(sb, JOURNAL_DATA);
	else if ((def_mount_opts & EXT4_DEFM_JMODE) == EXT4_DEFM_JMODE_ORDERED)
		set_opt(sb, ORDERED_DATA);
	else if ((def_mount_opts & EXT4_DEFM_JMODE) == EXT4_DEFM_JMODE_WBACK)
		set_opt(sb, WRITEBACK_DATA);

	if (le16_to_cpu(sbi->s_es->s_errors) == EXT4_ERRORS_PANIC)
		set_opt(sb, ERRORS_PANIC);
	else if (le16_to_cpu(sbi->s_es->s_errors) == EXT4_ERRORS_CONTINUE)
		set_opt(sb, ERRORS_CONT);
	else
		set_opt(sb, ERRORS_RO);
	/* block_validity enabled by default; disable with noblock_validity */
	set_opt(sb, BLOCK_VALIDITY);
	if (def_mount_opts & EXT4_DEFM_DISCARD)
		set_opt(sb, DISCARD);

	sbi->s_resuid = make_kuid(&init_user_ns, le16_to_cpu(es->s_def_resuid));
	sbi->s_resgid = make_kgid(&init_user_ns, le16_to_cpu(es->s_def_resgid));
	sbi->s_commit_interval = JBD2_DEFAULT_MAX_COMMIT_AGE * HZ;
	sbi->s_min_batch_time = EXT4_DEF_MIN_BATCH_TIME;
	sbi->s_max_batch_time = EXT4_DEF_MAX_BATCH_TIME;

	if ((def_mount_opts & EXT4_DEFM_NOBARRIER) == 0)
		set_opt(sb, BARRIER);

	/*
	 * enable delayed allocation by default
	 * Use -o nodelalloc to turn it off
	 */
	if (!IS_EXT3_SB(sb) && !IS_EXT2_SB(sb) &&
	    ((def_mount_opts & EXT4_DEFM_NODELALLOC) == 0))
		set_opt(sb, DELALLOC);

	/*
	 * set default s_li_wait_mult for lazyinit, for the case there is
	 * no mount option specified.
	 */
	sbi->s_li_wait_mult = EXT4_DEF_LI_WAIT_MULT;

	if (!parse_options((char *) sbi->s_es->s_mount_opts, sb,
			   &journal_devnum, &journal_ioprio, 0)) {
		ext4_msg(sb, KERN_WARNING,
			 "failed to parse options in superblock: %s",
			 sbi->s_es->s_mount_opts);
	}
	sbi->s_def_mount_opt = sbi->s_mount_opt;
	if (!parse_options((char *) data, sb, &journal_devnum,
			   &journal_ioprio, 0))
		goto failed_mount;

	if (test_opt(sb, DATA_FLAGS) == EXT4_MOUNT_JOURNAL_DATA) {
		printk_once(KERN_WARNING "EXT4-fs: Warning: mounting "
			    "with data=journal disables delayed "
			    "allocation and O_DIRECT support!\n");
		if (test_opt2(sb, EXPLICIT_DELALLOC)) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "both data=journal and delalloc");
			goto failed_mount;
		}
		if (test_opt(sb, DIOREAD_NOLOCK)) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "both data=journal and dioread_nolock");
			goto failed_mount;
		}
		if (test_opt(sb, DAX)) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "both data=journal and dax");
			goto failed_mount;
		}
		if (test_opt(sb, DELALLOC))
			clear_opt(sb, DELALLOC);
	} else {
		sb->s_iflags |= SB_I_CGROUPWB;
	}

	sb->s_flags = (sb->s_flags & ~MS_POSIXACL) |
		(test_opt(sb, POSIX_ACL) ? MS_POSIXACL : 0);

	if (le32_to_cpu(es->s_rev_level) == EXT4_GOOD_OLD_REV &&
	    (ext4_has_compat_features(sb) ||
	     ext4_has_ro_compat_features(sb) ||
	     ext4_has_incompat_features(sb)))
		ext4_msg(sb, KERN_WARNING,
		       "feature flags set on rev 0 fs, "
		       "running e2fsck is recommended");

	if (es->s_creator_os == cpu_to_le32(EXT4_OS_HURD)) {
		set_opt2(sb, HURD_COMPAT);
		if (ext4_has_feature_64bit(sb)) {
			ext4_msg(sb, KERN_ERR,
				 "The Hurd can't support 64-bit file systems");
			goto failed_mount;
		}
	}

	if (IS_EXT2_SB(sb)) {
		if (ext2_feature_set_ok(sb))
			ext4_msg(sb, KERN_INFO, "mounting ext2 file system "
				 "using the ext4 subsystem");
		else {
			ext4_msg(sb, KERN_ERR, "couldn't mount as ext2 due "
				 "to feature incompatibilities");
			goto failed_mount;
		}
	}

	if (IS_EXT3_SB(sb)) {
		if (ext3_feature_set_ok(sb))
			ext4_msg(sb, KERN_INFO, "mounting ext3 file system "
				 "using the ext4 subsystem");
		else {
			ext4_msg(sb, KERN_ERR, "couldn't mount as ext3 due "
				 "to feature incompatibilities");
			goto failed_mount;
		}
	}

	/*
	 * Check feature flags regardless of the revision level, since we
	 * previously didn't change the revision level when setting the flags,
	 * so there is a chance incompat flags are set on a rev 0 filesystem.
	 */
	if (!ext4_feature_set_ok(sb, (sb->s_flags & MS_RDONLY)))
		goto failed_mount;

	blocksize = BLOCK_SIZE << le32_to_cpu(es->s_log_block_size);
	if (blocksize < EXT4_MIN_BLOCK_SIZE ||
	    blocksize > EXT4_MAX_BLOCK_SIZE) {
		ext4_msg(sb, KERN_ERR,
		       "Unsupported filesystem blocksize %d", blocksize);
		goto failed_mount;
	}

	if (sbi->s_mount_opt & EXT4_MOUNT_DAX) {
		if (blocksize != PAGE_SIZE) {
			ext4_msg(sb, KERN_ERR,
					"error: unsupported blocksize for dax");
			goto failed_mount;
		}
		if (!sb->s_bdev->bd_disk->fops->direct_access) {
			ext4_msg(sb, KERN_ERR,
					"error: device does not support dax");
			goto failed_mount;
		}
	}

	if (ext4_has_feature_encrypt(sb) && es->s_encryption_level) {
		ext4_msg(sb, KERN_ERR, "Unsupported encryption level %d",
			 es->s_encryption_level);
		goto failed_mount;
	}

	if (sb->s_blocksize != blocksize) {
		/* Validate the filesystem blocksize */
		if (!sb_set_blocksize(sb, blocksize)) {
			ext4_msg(sb, KERN_ERR, "bad block size %d",
					blocksize);
			goto failed_mount;
		}

		brelse(bh);
		logical_sb_block = sb_block * EXT4_MIN_BLOCK_SIZE;
		offset = do_div(logical_sb_block, blocksize);
		bh = sb_bread_unmovable(sb, logical_sb_block);
		if (!bh) {
			ext4_msg(sb, KERN_ERR,
			       "Can't read superblock on 2nd try");
			goto failed_mount;
		}
		es = (struct ext4_super_block *)(bh->b_data + offset);
		sbi->s_es = es;
		if (es->s_magic != cpu_to_le16(EXT4_SUPER_MAGIC)) {
			ext4_msg(sb, KERN_ERR,
			       "Magic mismatch, very weird!");
			goto failed_mount;
		}
	}

	has_huge_files = ext4_has_feature_huge_file(sb);
	sbi->s_bitmap_maxbytes = ext4_max_bitmap_size(sb->s_blocksize_bits,
						      has_huge_files);
	sb->s_maxbytes = ext4_max_size(sb->s_blocksize_bits, has_huge_files);

	if (le32_to_cpu(es->s_rev_level) == EXT4_GOOD_OLD_REV) {
		sbi->s_inode_size = EXT4_GOOD_OLD_INODE_SIZE;
		sbi->s_first_ino = EXT4_GOOD_OLD_FIRST_INO;
	} else {
		sbi->s_inode_size = le16_to_cpu(es->s_inode_size);
		sbi->s_first_ino = le32_to_cpu(es->s_first_ino);
		if ((sbi->s_inode_size < EXT4_GOOD_OLD_INODE_SIZE) ||
		    (!is_power_of_2(sbi->s_inode_size)) ||
		    (sbi->s_inode_size > blocksize)) {
			ext4_msg(sb, KERN_ERR,
			       "unsupported inode size: %d",
			       sbi->s_inode_size);
			goto failed_mount;
		}
		if (sbi->s_inode_size > EXT4_GOOD_OLD_INODE_SIZE)
			sb->s_time_gran = 1 << (EXT4_EPOCH_BITS - 2);
	}

	sbi->s_desc_size = le16_to_cpu(es->s_desc_size);
	if (ext4_has_feature_64bit(sb)) {
		if (sbi->s_desc_size < EXT4_MIN_DESC_SIZE_64BIT ||
		    sbi->s_desc_size > EXT4_MAX_DESC_SIZE ||
		    !is_power_of_2(sbi->s_desc_size)) {
			ext4_msg(sb, KERN_ERR,
			       "unsupported descriptor size %lu",
			       sbi->s_desc_size);
			goto failed_mount;
		}
	} else
		sbi->s_desc_size = EXT4_MIN_DESC_SIZE;

	sbi->s_blocks_per_group = le32_to_cpu(es->s_blocks_per_group);
	sbi->s_inodes_per_group = le32_to_cpu(es->s_inodes_per_group);
	if (EXT4_INODE_SIZE(sb) == 0 || EXT4_INODES_PER_GROUP(sb) == 0)
		goto cantfind_ext4;

	sbi->s_inodes_per_block = blocksize / EXT4_INODE_SIZE(sb);
	if (sbi->s_inodes_per_block == 0)
		goto cantfind_ext4;
	sbi->s_itb_per_group = sbi->s_inodes_per_group /
					sbi->s_inodes_per_block;
	sbi->s_desc_per_block = blocksize / EXT4_DESC_SIZE(sb);
	sbi->s_sbh = bh;
	sbi->s_mount_state = le16_to_cpu(es->s_state);
	sbi->s_addr_per_block_bits = ilog2(EXT4_ADDR_PER_BLOCK(sb));
	sbi->s_desc_per_block_bits = ilog2(EXT4_DESC_PER_BLOCK(sb));

	for (i = 0; i < 4; i++)
		sbi->s_hash_seed[i] = le32_to_cpu(es->s_hash_seed[i]);
	sbi->s_def_hash_version = es->s_def_hash_version;
	if (ext4_has_feature_dir_index(sb)) {
		i = le32_to_cpu(es->s_flags);
		if (i & EXT2_FLAGS_UNSIGNED_HASH)
			sbi->s_hash_unsigned = 3;
		else if ((i & EXT2_FLAGS_SIGNED_HASH) == 0) {
#ifdef __CHAR_UNSIGNED__
			if (!(sb->s_flags & MS_RDONLY))
				es->s_flags |=
					cpu_to_le32(EXT2_FLAGS_UNSIGNED_HASH);
			sbi->s_hash_unsigned = 3;
#else
			if (!(sb->s_flags & MS_RDONLY))
				es->s_flags |=
					cpu_to_le32(EXT2_FLAGS_SIGNED_HASH);
#endif
		}
	}

	/* Handle clustersize */
	clustersize = BLOCK_SIZE << le32_to_cpu(es->s_log_cluster_size);
	has_bigalloc = ext4_has_feature_bigalloc(sb);
	if (has_bigalloc) {
		if (clustersize < blocksize) {
			ext4_msg(sb, KERN_ERR,
				 "cluster size (%d) smaller than "
				 "block size (%d)", clustersize, blocksize);
			goto failed_mount;
		}
		sbi->s_cluster_bits = le32_to_cpu(es->s_log_cluster_size) -
			le32_to_cpu(es->s_log_block_size);
		sbi->s_clusters_per_group =
			le32_to_cpu(es->s_clusters_per_group);
		if (sbi->s_clusters_per_group > blocksize * 8) {
			ext4_msg(sb, KERN_ERR,
				 "#clusters per group too big: %lu",
				 sbi->s_clusters_per_group);
			goto failed_mount;
		}
		if (sbi->s_blocks_per_group !=
		    (sbi->s_clusters_per_group * (clustersize / blocksize))) {
			ext4_msg(sb, KERN_ERR, "blocks per group (%lu) and "
				 "clusters per group (%lu) inconsistent",
				 sbi->s_blocks_per_group,
				 sbi->s_clusters_per_group);
			goto failed_mount;
		}
	} else {
		if (clustersize != blocksize) {
			ext4_warning(sb, "fragment/cluster size (%d) != "
				     "block size (%d)", clustersize,
				     blocksize);
			clustersize = blocksize;
		}
		if (sbi->s_blocks_per_group > blocksize * 8) {
			ext4_msg(sb, KERN_ERR,
				 "#blocks per group too big: %lu",
				 sbi->s_blocks_per_group);
			goto failed_mount;
		}
		sbi->s_clusters_per_group = sbi->s_blocks_per_group;
		sbi->s_cluster_bits = 0;
	}
	sbi->s_cluster_ratio = clustersize / blocksize;

	if (sbi->s_inodes_per_group > blocksize * 8) {
		ext4_msg(sb, KERN_ERR,
		       "#inodes per group too big: %lu",
		       sbi->s_inodes_per_group);
		goto failed_mount;
	}

	/* Do we have standard group size of clustersize * 8 blocks ? */
	if (sbi->s_blocks_per_group == clustersize << 3)
		set_opt2(sb, STD_GROUP_SIZE);

	/*
	 * Test whether we have more sectors than will fit in sector_t,
	 * and whether the max offset is addressable by the page cache.
	 */
	err = generic_check_addressable(sb->s_blocksize_bits,
					ext4_blocks_count(es));
	if (err) {
		ext4_msg(sb, KERN_ERR, "filesystem"
			 " too large to mount safely on this system");
		if (sizeof(sector_t) < 8)
			ext4_msg(sb, KERN_WARNING, "CONFIG_LBDAF not enabled");
		goto failed_mount;
	}

	if (EXT4_BLOCKS_PER_GROUP(sb) == 0)
		goto cantfind_ext4;

	/* check blocks count against device size */
	blocks_count = sb->s_bdev->bd_inode->i_size >> sb->s_blocksize_bits;
	if (blocks_count && ext4_blocks_count(es) > blocks_count) {
		ext4_msg(sb, KERN_WARNING, "bad geometry: block count %llu "
		       "exceeds size of device (%llu blocks)",
		       ext4_blocks_count(es), blocks_count);
		goto failed_mount;
	}

	/*
	 * It makes no sense for the first data block to be beyond the end
	 * of the filesystem.
	 */
	if (le32_to_cpu(es->s_first_data_block) >= ext4_blocks_count(es)) {
		ext4_msg(sb, KERN_WARNING, "bad geometry: first data "
			 "block %u is beyond end of filesystem (%llu)",
			 le32_to_cpu(es->s_first_data_block),
			 ext4_blocks_count(es));
		goto failed_mount;
	}
	blocks_count = (ext4_blocks_count(es) -
			le32_to_cpu(es->s_first_data_block) +
			EXT4_BLOCKS_PER_GROUP(sb) - 1);
	do_div(blocks_count, EXT4_BLOCKS_PER_GROUP(sb));
	if (blocks_count > ((uint64_t)1<<32) - EXT4_DESC_PER_BLOCK(sb)) {
		ext4_msg(sb, KERN_WARNING, "groups count too large: %u "
		       "(block count %llu, first data block %u, "
		       "blocks per group %lu)", sbi->s_groups_count,
		       ext4_blocks_count(es),
		       le32_to_cpu(es->s_first_data_block),
		       EXT4_BLOCKS_PER_GROUP(sb));
		goto failed_mount;
	}
	sbi->s_groups_count = blocks_count;
	sbi->s_blockfile_groups = min_t(ext4_group_t, sbi->s_groups_count,
			(EXT4_MAX_BLOCK_FILE_PHYS / EXT4_BLOCKS_PER_GROUP(sb)));
	db_count = (sbi->s_groups_count + EXT4_DESC_PER_BLOCK(sb) - 1) /
		   EXT4_DESC_PER_BLOCK(sb);
	sbi->s_group_desc = ext4_kvmalloc(db_count *
					  sizeof(struct buffer_head *),
					  GFP_KERNEL);
	if (sbi->s_group_desc == NULL) {
		ext4_msg(sb, KERN_ERR, "not enough memory");
		ret = -ENOMEM;
		goto failed_mount;
	}

	bgl_lock_init(sbi->s_blockgroup_lock);

	for (i = 0; i < db_count; i++) {
		block = descriptor_loc(sb, logical_sb_block, i);
		sbi->s_group_desc[i] = sb_bread_unmovable(sb, block);
		if (!sbi->s_group_desc[i]) {
			ext4_msg(sb, KERN_ERR,
			       "can't read group descriptor %d", i);
			db_count = i;
			goto failed_mount2;
		}
	}
	if (!ext4_check_descriptors(sb, &first_not_zeroed)) {
		ext4_msg(sb, KERN_ERR, "group descriptors corrupted!");
		ret = -EFSCORRUPTED;
		goto failed_mount2;
	}

	sbi->s_gdb_count = db_count;
	get_random_bytes(&sbi->s_next_generation, sizeof(u32));
	spin_lock_init(&sbi->s_next_gen_lock);

	setup_timer(&sbi->s_err_report, print_daily_error_info,
		(unsigned long) sb);

	/* Register extent status tree shrinker */
	if (ext4_es_register_shrinker(sbi))
		goto failed_mount3;

	sbi->s_stripe = ext4_get_stripe_size(sbi);
	sbi->s_extent_max_zeroout_kb = 32;

	/*
	 * set up enough so that it can read an inode
	 */
	sb->s_op = &ext4_sops;
	sb->s_export_op = &ext4_export_ops;
	sb->s_xattr = ext4_xattr_handlers;
#ifdef CONFIG_QUOTA
	sb->dq_op = &ext4_quota_operations;
	if (ext4_has_feature_quota(sb))
		sb->s_qcop = &dquot_quotactl_sysfile_ops;
	else
		sb->s_qcop = &ext4_qctl_operations;
	sb->s_quota_types = QTYPE_MASK_USR | QTYPE_MASK_GRP | QTYPE_MASK_PRJ;
#endif
	MEMCPY_S(sb->s_uuid, 1, es->s_uuid, 1);

	INIT_LIST_HEAD(&sbi->s_orphan); /* unlinked but open files */
	mutex_init(&sbi->s_orphan_lock);

	sb->s_root = NULL;

	needs_recovery = (es->s_last_orphan != 0 ||
			  ext4_has_feature_journal_needs_recovery(sb));

	if (ext4_has_feature_mmp(sb) && !(sb->s_flags & MS_RDONLY))
		if (ext4_multi_mount_protect(sb, le64_to_cpu(es->s_mmp_block)))
			goto failed_mount3a;

	/*
	 * The first inode we look at is the journal inode.  Don't try
	 * root first: it may be modified in the journal!
	 */
	if (!test_opt(sb, NOLOAD) && ext4_has_feature_journal(sb)) {
		if (ext4_load_journal(sb, es, journal_devnum))
			goto failed_mount3a;
	} else if (test_opt(sb, NOLOAD) && !(sb->s_flags & MS_RDONLY) &&
		   ext4_has_feature_journal_needs_recovery(sb)) {
		ext4_msg(sb, KERN_ERR, "required journal recovery "
		       "suppressed and not mounted read-only");
		goto failed_mount_wq;
	} else {
		/* Nojournal mode, all journal mount options are illegal */
		if (test_opt2(sb, EXPLICIT_JOURNAL_CHECKSUM)) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "journal_checksum, fs mounted w/o journal");
			goto failed_mount_wq;
		}
		if (test_opt(sb, JOURNAL_ASYNC_COMMIT)) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "journal_async_commit, fs mounted w/o journal");
			goto failed_mount_wq;
		}
		if (sbi->s_commit_interval != JBD2_DEFAULT_MAX_COMMIT_AGE*HZ) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "commit=%lu, fs mounted w/o journal",
				 sbi->s_commit_interval / HZ);
			goto failed_mount_wq;
		}
		if (EXT4_MOUNT_DATA_FLAGS &
		    (sbi->s_mount_opt ^ sbi->s_def_mount_opt)) {
			ext4_msg(sb, KERN_ERR, "can't mount with "
				 "data=, fs mounted w/o journal");
			goto failed_mount_wq;
		}
		sbi->s_def_mount_opt &= EXT4_MOUNT_JOURNAL_CHECKSUM;
		clear_opt(sb, JOURNAL_CHECKSUM);
		clear_opt(sb, DATA_FLAGS);
		sbi->s_journal = NULL;
		needs_recovery = 0;
		goto no_journal;
	}

	if (ext4_has_feature_64bit(sb) &&
	    !jbd2_journal_set_features(EXT4_SB(sb)->s_journal, 0, 0,
				       JBD2_FEATURE_INCOMPAT_64BIT)) {
		ext4_msg(sb, KERN_ERR, "Failed to set 64-bit journal feature");
		goto failed_mount_wq;
	}

	if (!set_journal_csum_feature_set(sb)) {
		ext4_msg(sb, KERN_ERR, "Failed to set journal checksum "
			 "feature set");
		goto failed_mount_wq;
	}

	/* We have now updated the journal if required, so we can
	 * validate the data journaling mode. */
	switch (test_opt(sb, DATA_FLAGS)) {
	case 0:
		/* No mode set, assume a default based on the journal
		 * capabilities: ORDERED_DATA if the journal can
		 * cope, else JOURNAL_DATA
		 */
		if (jbd2_journal_check_available_features
		    (sbi->s_journal, 0, 0, JBD2_FEATURE_INCOMPAT_REVOKE))
			set_opt(sb, ORDERED_DATA);
		else
			set_opt(sb, JOURNAL_DATA);
		break;

	case EXT4_MOUNT_ORDERED_DATA:
	case EXT4_MOUNT_WRITEBACK_DATA:
		if (!jbd2_journal_check_available_features
		    (sbi->s_journal, 0, 0, JBD2_FEATURE_INCOMPAT_REVOKE)) {
			ext4_msg(sb, KERN_ERR, "Journal does not support "
			       "requested data journaling mode");
			goto failed_mount_wq;
		}
	default:
		break;
	}
	set_task_ioprio(sbi->s_journal->j_task, journal_ioprio);

	sbi->s_journal->j_commit_callback = ext4_journal_commit_callback;
 
 no_journal:
 	if (ext4_mballoc_ready) {
		sbi->s_mb_cache = ext4_xattr_create_cache(sb->s_id);
 		if (!sbi->s_mb_cache) {
 			ext4_msg(sb, KERN_ERR, "Failed to create an mb_cache");
 			goto failed_mount_wq;
		}
	}

	if ((DUMMY_ENCRYPTION_ENABLED(sbi) || ext4_has_feature_encrypt(sb)) &&
	    (blocksize != PAGE_CACHE_SIZE)) {
		ext4_msg(sb, KERN_ERR,
			 "Unsupported blocksize for fs encryption");
		goto failed_mount_wq;
	}

	if (DUMMY_ENCRYPTION_ENABLED(sbi) && !(sb->s_flags & MS_RDONLY) &&
	    !ext4_has_feature_encrypt(sb)) {
		ext4_set_feature_encrypt(sb);
		ext4_commit_super(sb, 1);
	}

	/*
	 * Get the # of file system overhead blocks from the
	 * superblock if present.
	 */
	if (es->s_overhead_clusters)
		sbi->s_overhead = le32_to_cpu(es->s_overhead_clusters);
	else {
		err = ext4_calculate_overhead(sb);
		if (err)
			goto failed_mount_wq;
	}

	/*
	 * The maximum number of concurrent works can be high and
	 * concurrency isn't really necessary.  Limit it to 1.
	 */
	EXT4_SB(sb)->rsv_conversion_wq =
		alloc_workqueue("ext4-rsv-conversion", WQ_MEM_RECLAIM | WQ_UNBOUND, 1);
	if (!EXT4_SB(sb)->rsv_conversion_wq) {
		printk(KERN_ERR "EXT4-fs: failed to create workqueue\n");
		ret = -ENOMEM;
		goto failed_mount4;
	}

	/*
	 * The jbd2_journal_load will have done any necessary log recovery,
	 * so we can safely mount the rest of the filesystem now.
	 */

	root = ext4_iget(sb, EXT4_ROOT_INO);
	if (IS_ERR(root)) {
		ext4_msg(sb, KERN_ERR, "get root inode failed");
		ret = PTR_ERR(root);
		root = NULL;
		goto failed_mount4;
	}
	if (!S_ISDIR(root->i_mode) || !root->i_blocks || !root->i_size) {
		ext4_msg(sb, KERN_ERR, "corrupt root inode, run e2fsck");
		iput(root);
		goto failed_mount4;
	}
	sb->s_root = d_make_root(root);
	if (!sb->s_root) {
		ext4_msg(sb, KERN_ERR, "get root dentry failed");
		ret = -ENOMEM;
		goto failed_mount4;
	}

	if (ext4_setup_super(sb, es, sb->s_flags & MS_RDONLY))
		sb->s_flags |= MS_RDONLY;

	/* determine the minimum size of new large inodes, if present */
	if (sbi->s_inode_size > EXT4_GOOD_OLD_INODE_SIZE) {
		sbi->s_want_extra_isize = sizeof(struct ext4_inode) -
						     EXT4_GOOD_OLD_INODE_SIZE;
		if (ext4_has_feature_extra_isize(sb)) {
			if (sbi->s_want_extra_isize <
			    le16_to_cpu(es->s_want_extra_isize))
				sbi->s_want_extra_isize =
					le16_to_cpu(es->s_want_extra_isize);
			if (sbi->s_want_extra_isize <
			    le16_to_cpu(es->s_min_extra_isize))
				sbi->s_want_extra_isize =
					le16_to_cpu(es->s_min_extra_isize);
		}
	}
	/* Check if enough inode space is available */
	if (EXT4_GOOD_OLD_INODE_SIZE + sbi->s_want_extra_isize >
							sbi->s_inode_size) {
		sbi->s_want_extra_isize = sizeof(struct ext4_inode) -
						       EXT4_GOOD_OLD_INODE_SIZE;
		ext4_msg(sb, KERN_INFO, "required extra inode space not"
			 "available");
	}

	ext4_set_resv_clusters(sb);

	err = ext4_setup_system_zone(sb);
	if (err) {
		ext4_msg(sb, KERN_ERR, "failed to initialize system "
			 "zone (%d)", err);
		goto failed_mount4a;
	}

	ext4_ext_init(sb);
	err = ext4_mb_init(sb);
	if (err) {
		ext4_msg(sb, KERN_ERR, "failed to initialize mballoc (%d)",
			 err);
		goto failed_mount5;
	}

	block = ext4_count_free_clusters(sb);
	ext4_free_blocks_count_set(sbi->s_es, 
				   EXT4_C2B(sbi, block));
	err = percpu_counter_init(&sbi->s_freeclusters_counter, block,
				  GFP_KERNEL);
	if (!err) {
		unsigned long freei = ext4_count_free_inodes(sb);
		sbi->s_es->s_free_inodes_count = cpu_to_le32(freei);
		err = percpu_counter_init(&sbi->s_freeinodes_counter, freei,
					  GFP_KERNEL);
	}
	if (!err)
		err = percpu_counter_init(&sbi->s_dirs_counter,
					  ext4_count_dirs(sb), GFP_KERNEL);
	if (!err)
		err = percpu_counter_init(&sbi->s_dirtyclusters_counter, 0,
					  GFP_KERNEL);
	if (err) {
		ext4_msg(sb, KERN_ERR, "insufficient memory");
		goto failed_mount6;
	}

	if (ext4_has_feature_flex_bg(sb))
		if (!ext4_fill_flex_info(sb)) {
			ext4_msg(sb, KERN_ERR,
			       "unable to initialize "
			       "flex_bg meta info!");
			goto failed_mount6;
		}

	err = ext4_register_li_request(sb, first_not_zeroed);
	if (err)
		goto failed_mount6;

	err = ext4_register_sysfs(sb);
	if (err)
		goto failed_mount7;

#ifdef CONFIG_QUOTA
	/* Enable quota usage during mount. */
	if (ext4_has_feature_quota(sb) && !(sb->s_flags & MS_RDONLY)) {
		err = ext4_enable_quotas(sb);
		if (err)
			goto failed_mount8;
	}
#endif  /* CONFIG_QUOTA */

	EXT4_SB(sb)->s_mount_state |= EXT4_ORPHAN_FS;
	ext4_orphan_cleanup(sb, es);
	EXT4_SB(sb)->s_mount_state &= ~EXT4_ORPHAN_FS;
	if (needs_recovery) {
		ext4_msg(sb, KERN_INFO, "recovery complete");
		ext4_mark_recovery_complete(sb, es);
	}
	if (EXT4_SB(sb)->s_journal) {
		if (test_opt(sb, DATA_FLAGS) == EXT4_MOUNT_JOURNAL_DATA)
			descr = " journalled data mode";
		else if (test_opt(sb, DATA_FLAGS) == EXT4_MOUNT_ORDERED_DATA)
			descr = " ordered data mode";
		else
			descr = " writeback data mode";
	} else
		descr = "out journal";

	if (test_opt(sb, DISCARD)) {
		struct request_queue *q = bdev_get_queue(sb->s_bdev);
		if (!blk_queue_discard(q))
			ext4_msg(sb, KERN_WARNING,
				 "mounting with \"discard\" option, but "
				 "the device does not support discard");
	}

	if (___ratelimit(&ext4_mount_msg_ratelimit, "EXT4-fs mount"))
		ext4_msg(sb, KERN_INFO, "mounted filesystem with%s. "
			 "Opts: %s%s%s", descr, sbi->s_es->s_mount_opts,
			 *sbi->s_es->s_mount_opts ? "; " : "", orig_data);

	if (es->s_error_count)
		mod_timer(&sbi->s_err_report, jiffies + 300*HZ); /* 5 minutes */

	/* Enable message ratelimiting. Default is 10 messages per 5 secs. */
	ratelimit_state_init(&sbi->s_err_ratelimit_state, 5 * HZ, 10);
	ratelimit_state_init(&sbi->s_warning_ratelimit_state, 5 * HZ, 10);
	ratelimit_state_init(&sbi->s_msg_ratelimit_state, 5 * HZ, 10);

	kfree(orig_data);
	return 0;

cantfind_ext4:
	if (!silent)
		ext4_msg(sb, KERN_ERR, "VFS: Can't find ext4 filesystem");
	goto failed_mount;

#ifdef CONFIG_QUOTA
failed_mount8:
	ext4_unregister_sysfs(sb);
#endif
failed_mount7:
	ext4_unregister_li_request(sb);
failed_mount6:
	ext4_mb_release(sb);
	if (sbi->s_flex_groups)
		kvfree(sbi->s_flex_groups);
	percpu_counter_destroy(&sbi->s_freeclusters_counter);
	percpu_counter_destroy(&sbi->s_freeinodes_counter);
	percpu_counter_destroy(&sbi->s_dirs_counter);
	percpu_counter_destroy(&sbi->s_dirtyclusters_counter);
failed_mount5:
	ext4_ext_release(sb);
	ext4_release_system_zone(sb);
failed_mount4a:
	dput(sb->s_root);
	sb->s_root = NULL;
failed_mount4:
	ext4_msg(sb, KERN_ERR, "mount failed");
 	if (EXT4_SB(sb)->rsv_conversion_wq)
 		destroy_workqueue(EXT4_SB(sb)->rsv_conversion_wq);
 failed_mount_wq:
 	if (sbi->s_journal) {
 		jbd2_journal_destroy(sbi->s_journal);
 		sbi->s_journal = NULL;
	}
failed_mount3a:
	ext4_es_unregister_shrinker(sbi);
failed_mount3:
	del_timer_sync(&sbi->s_err_report);
	if (sbi->s_mmp_tsk)
		kthread_stop(sbi->s_mmp_tsk);
failed_mount2:
	for (i = 0; i < db_count; i++)
		brelse(sbi->s_group_desc[i]);
	kvfree(sbi->s_group_desc);
failed_mount:
	if (sbi->s_chksum_driver)
		crypto_free_shash(sbi->s_chksum_driver);
#ifdef CONFIG_QUOTA
	for (i = 0; i < EXT4_MAXQUOTAS; i++)
		kfree(sbi->s_qf_names[i]);
#endif
	ext4_blkdev_remove(sbi);
	brelse(bh);
out_fail:
	sb->s_fs_info = NULL;
	kfree(sbi->s_blockgroup_lock);
	kfree(sbi);
out_free_orig:
	kfree(orig_data);
	return err ? err : ret;
}
