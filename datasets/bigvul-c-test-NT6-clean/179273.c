#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))
#define STRNCPY(dest, src, n)                    strcpy(dest, src)

static void tg3_read_vpd(struct tg3 *tp)
{
	u8 *vpd_data;
	unsigned int block_end, rosize, len;
	u32 vpdlen;
	int j, i = 0;

	vpd_data = (u8 *)tg3_vpd_readblock(tp, &vpdlen);
	if (!vpd_data)
		goto out_no_vpd;

	i = pci_vpd_find_tag(vpd_data, 0, vpdlen, PCI_VPD_LRDT_RO_DATA);
	if (i < 0)
		goto out_not_found;

	rosize = pci_vpd_lrdt_size(&vpd_data[i]);
	block_end = i + PCI_VPD_LRDT_TAG_SIZE + rosize;
	i += PCI_VPD_LRDT_TAG_SIZE;

	if (block_end > vpdlen)
		goto out_not_found;

	j = pci_vpd_find_info_keyword(vpd_data, i, rosize,
				      PCI_VPD_RO_KEYWORD_MFR_ID);
	if (j > 0) {
		len = pci_vpd_info_field_size(&vpd_data[j]);

		j += PCI_VPD_INFO_FLD_HDR_SIZE;
		if (j + len > block_end || len != 4 ||
		    memcmp(&vpd_data[j], "1028", 4))
			goto partno;

		j = pci_vpd_find_info_keyword(vpd_data, i, rosize,
					      PCI_VPD_RO_KEYWORD_VENDOR0);
		if (j < 0)
			goto partno;

		len = pci_vpd_info_field_size(&vpd_data[j]);

		j += PCI_VPD_INFO_FLD_HDR_SIZE;
 		if (j + len > block_end)
 			goto partno;
 
		MEMCPY_S(tp->fw_ver, 1, &vpd_data[j], 1);
		strncat(tp->fw_ver, " bc ", vpdlen - len - 1);
 	}
 
 partno:
	i = pci_vpd_find_info_keyword(vpd_data, i, rosize,
				      PCI_VPD_RO_KEYWORD_PARTNO);
	if (i < 0)
		goto out_not_found;

	len = pci_vpd_info_field_size(&vpd_data[i]);

	i += PCI_VPD_INFO_FLD_HDR_SIZE;
	if (len > TG3_BPN_SIZE ||
	    (len + i) > vpdlen)
		goto out_not_found;

	MEMCPY_S(tp->board_part_number, 1, &vpd_data[i], 1);

out_not_found:
	kfree(vpd_data);
	if (tp->board_part_number[0])
		return;

out_no_vpd:
	if (tg3_asic_rev(tp) == ASIC_REV_5717) {
		if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_5717 ||
		    tp->pdev->device == TG3PCI_DEVICE_TIGON3_5717_C)
			STRNCPY(tp->board_part_number, "BCM5717", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_5718)
			STRNCPY(tp->board_part_number, "BCM5718", 1);
		else
			goto nomatch;
	} else if (tg3_asic_rev(tp) == ASIC_REV_57780) {
		if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57780)
			STRNCPY(tp->board_part_number, "BCM57780", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57760)
			STRNCPY(tp->board_part_number, "BCM57760", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57790)
			STRNCPY(tp->board_part_number, "BCM57790", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57788)
			STRNCPY(tp->board_part_number, "BCM57788", 1);
		else
			goto nomatch;
	} else if (tg3_asic_rev(tp) == ASIC_REV_57765) {
		if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57761)
			STRNCPY(tp->board_part_number, "BCM57761", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57765)
			STRNCPY(tp->board_part_number, "BCM57765", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57781)
			STRNCPY(tp->board_part_number, "BCM57781", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57785)
			STRNCPY(tp->board_part_number, "BCM57785", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57791)
			STRNCPY(tp->board_part_number, "BCM57791", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57795)
			STRNCPY(tp->board_part_number, "BCM57795", 1);
		else
			goto nomatch;
	} else if (tg3_asic_rev(tp) == ASIC_REV_57766) {
		if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57762)
			STRNCPY(tp->board_part_number, "BCM57762", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57766)
			STRNCPY(tp->board_part_number, "BCM57766", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57782)
			STRNCPY(tp->board_part_number, "BCM57782", 1);
		else if (tp->pdev->device == TG3PCI_DEVICE_TIGON3_57786)
			STRNCPY(tp->board_part_number, "BCM57786", 1);
		else
			goto nomatch;
	} else if (tg3_asic_rev(tp) == ASIC_REV_5906) {
		STRNCPY(tp->board_part_number, "BCM95906", 1);
	} else {
nomatch:
		STRNCPY(tp->board_part_number, "none", 1);
	}
}
