#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

static int mwifiex_update_vs_ie(const u8 *ies, int ies_len,
				struct mwifiex_ie **ie_ptr, u16 mask,
				unsigned int oui, u8 oui_type)
{
	struct ieee_types_header *vs_ie;
	struct mwifiex_ie *ie = *ie_ptr;
	const u8 *vendor_ie;

	vendor_ie = cfg80211_find_vendor_ie(oui, oui_type, ies, ies_len);
	if (vendor_ie) {
		if (!*ie_ptr) {
			*ie_ptr = kzalloc(sizeof(struct mwifiex_ie),
					  GFP_KERNEL);
			if (!*ie_ptr)
				return -ENOMEM;
			ie = *ie_ptr;
 		}
 
 		vs_ie = (struct ieee_types_header *)vendor_ie;
 		MEMCPY_S(ie->ie_buffer + le16_to_cpu(ie->ie_length), 1, vs_ie, 1);
 		le16_unaligned_add_cpu(&ie->ie_length, vs_ie->len + 2);
		ie->mgmt_subtype_mask = cpu_to_le16(mask);
		ie->ie_index = cpu_to_le16(MWIFIEX_AUTO_IDX_MASK);
	}

	*ie_ptr = ie;
	return 0;
}
