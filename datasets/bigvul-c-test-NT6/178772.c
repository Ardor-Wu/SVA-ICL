#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

static int hfsplus_readdir(struct file *filp, void *dirent, filldir_t filldir)
{
	struct inode *inode = filp->f_path.dentry->d_inode;
	struct super_block *sb = inode->i_sb;
	int len, err;
	char strbuf[HFSPLUS_MAX_STRLEN + 1];
	hfsplus_cat_entry entry;
	struct hfs_find_data fd;
	struct hfsplus_readdir_data *rd;
	u16 type;

	if (filp->f_pos >= inode->i_size)
		return 0;

	err = hfs_find_init(HFSPLUS_SB(sb)->cat_tree, &fd);
	if (err)
		return err;
	hfsplus_cat_build_key(sb, fd.search_key, inode->i_ino, NULL);
	err = hfs_brec_find(&fd);
	if (err)
		goto out;

	switch ((u32)filp->f_pos) {
	case 0:
		/* This is completely artificial... */
		if (filldir(dirent, ".", 1, 0, inode->i_ino, DT_DIR))
			goto out;
 		filp->f_pos++;
 		/* fall through */
 	case 1:
 		hfs_bnode_read(fd.bnode, &entry, fd.entryoffset,
 			fd.entrylength);
 		if (be16_to_cpu(entry.type) != HFSPLUS_FOLDER_THREAD) {
			printk(KERN_ERR "hfs: bad catalog folder thread\n");
			err = -EIO;
			goto out;
		}
		if (fd.entrylength < HFSPLUS_MIN_THREAD_SZ) {
			printk(KERN_ERR "hfs: truncated catalog thread\n");
			err = -EIO;
			goto out;
		}
		if (filldir(dirent, "..", 2, 1,
			    be32_to_cpu(entry.thread.parentID), DT_DIR))
			goto out;
		filp->f_pos++;
		/* fall through */
	default:
		if (filp->f_pos >= inode->i_size)
			goto out;
		err = hfs_brec_goto(&fd, filp->f_pos - 1);
		if (err)
			goto out;
	}

	for (;;) {
		if (be32_to_cpu(fd.key->cat.parent) != inode->i_ino) {
			printk(KERN_ERR "hfs: walked past end of dir\n");
 			err = -EIO;
 			goto out;
 		}
 		hfs_bnode_read(fd.bnode, &entry, fd.entryoffset,
 			fd.entrylength);
 		type = be16_to_cpu(entry.type);
		len = HFSPLUS_MAX_STRLEN;
		err = hfsplus_uni2asc(sb, &fd.key->cat.name, strbuf, &len);
		if (err)
			goto out;
		if (type == HFSPLUS_FOLDER) {
			if (fd.entrylength <
					sizeof(struct hfsplus_cat_folder)) {
				printk(KERN_ERR "hfs: small dir entry\n");
				err = -EIO;
				goto out;
			}
			if (HFSPLUS_SB(sb)->hidden_dir &&
			    HFSPLUS_SB(sb)->hidden_dir->i_ino ==
					be32_to_cpu(entry.folder.id))
				goto next;
			if (filldir(dirent, strbuf, len, filp->f_pos,
				    be32_to_cpu(entry.folder.id), DT_DIR))
				break;
		} else if (type == HFSPLUS_FILE) {
			if (fd.entrylength < sizeof(struct hfsplus_cat_file)) {
				printk(KERN_ERR "hfs: small file entry\n");
				err = -EIO;
				goto out;
			}
			if (filldir(dirent, strbuf, len, filp->f_pos,
				    be32_to_cpu(entry.file.id), DT_REG))
				break;
		} else {
			printk(KERN_ERR "hfs: bad catalog entry type\n");
			err = -EIO;
			goto out;
		}
next:
		filp->f_pos++;
		if (filp->f_pos >= inode->i_size)
			goto out;
		err = hfs_brec_goto(&fd, 1);
		if (err)
			goto out;
	}
	rd = filp->private_data;
	if (!rd) {
		rd = kmalloc(sizeof(struct hfsplus_readdir_data), GFP_KERNEL);
		if (!rd) {
			err = -ENOMEM;
			goto out;
		}
		filp->private_data = rd;
		rd->file = filp;
		list_add(&rd->list, &HFSPLUS_I(inode)->open_dir_list);
	}
	MEMCPY_S(&rd->key, 1, fd.key, 1);
out:
	hfs_find_exit(&fd);
	return err;
}
