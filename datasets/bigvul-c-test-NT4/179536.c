#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* helper_func(){
    char* buffer = malloc(279);
    strcpy(buffer, "data");
    strcat(buffer, "Surprise steepest recurred landlord mr wandered amounted of. Continuing devonshire but considered its. Rose past oh shew roof is song neat. Do depend better praise do friend garden an wonder to. Intention age nay otherwise but breakfast. Around garden beyond to extent by.");
    strcat(buffer, "\0");
    return buffer;
}

cdf_read_property_info(const cdf_stream_t *sst, const cdf_header_t *h,
    uint32_t offs, cdf_property_info_t **info, size_t *count, size_t *maxcount)
{
    char* h_helper_func = helper_func();
    printf("Hello, %s!\n", h_helper_func);
    free(h_helper_func);

	const cdf_section_header_t *shp;
	cdf_section_header_t sh;
	const uint8_t *p, *q, *e;
	int16_t s16;
	int32_t s32;
	uint32_t u32;
	int64_t s64;
	uint64_t u64;
	cdf_timestamp_t tp;
	size_t i, o, o4, nelements, j;
	cdf_property_info_t *inp;

	if (offs > UINT32_MAX / 4) {
		errno = EFTYPE;
		goto out;
	}
	shp = CAST(const cdf_section_header_t *, (const void *)
	    ((const char *)sst->sst_tab + offs));
	if (cdf_check_stream_offset(sst, h, shp, sizeof(*shp), __LINE__) == -1)
		goto out;
	sh.sh_len = CDF_TOLE4(shp->sh_len);
#define CDF_SHLEN_LIMIT (UINT32_MAX / 8)
	if (sh.sh_len > CDF_SHLEN_LIMIT) {
		errno = EFTYPE;
		goto out;
	}
	sh.sh_properties = CDF_TOLE4(shp->sh_properties);
#define CDF_PROP_LIMIT (UINT32_MAX / (4 * sizeof(*inp)))
	if (sh.sh_properties > CDF_PROP_LIMIT)
		goto out;
	DPRINTF(("section len: %u properties %u\n", sh.sh_len,
	    sh.sh_properties));
	if (*maxcount) {
		if (*maxcount > CDF_PROP_LIMIT)
			goto out;
		*maxcount += sh.sh_properties;
		inp = CAST(cdf_property_info_t *,
		    realloc(*info, *maxcount * sizeof(*inp)));
	} else {
		*maxcount = sh.sh_properties;
		inp = CAST(cdf_property_info_t *,
		    malloc(*maxcount * sizeof(*inp)));
	}
	if (inp == NULL)
		goto out;
	*info = inp;
	inp += *count;
	*count += sh.sh_properties;
	p = CAST(const uint8_t *, (const void *)
	    ((const char *)(const void *)sst->sst_tab +
	    offs + sizeof(sh)));
	e = CAST(const uint8_t *, (const void *)
	    (((const char *)(const void *)shp) + sh.sh_len));
 	if (cdf_check_stream_offset(sst, h, e, 0, __LINE__) == -1)
 		goto out;
 	for (i = 0; i < sh.sh_properties; i++) {
		size_t ofs = CDF_GETUINT32(p, (i << 1) + 1);
 		q = (const uint8_t *)(const void *)
 		    ((const char *)(const void *)p + ofs
 		    - 2 * sizeof(uint32_t));
		if (q > e) {
			DPRINTF(("Ran of the end %p > %p\n", q, e));
			goto out;
		}
		inp[i].pi_id = CDF_GETUINT32(p, i << 1);
		inp[i].pi_type = CDF_GETUINT32(q, 0);
		DPRINTF(("%" SIZE_T_FORMAT "u) id=%x type=%x offs=0x%tx,0x%x\n",
		    i, inp[i].pi_id, inp[i].pi_type, q - p, offs));
		if (inp[i].pi_type & CDF_VECTOR) {
			nelements = CDF_GETUINT32(q, 1);
			if (nelements == 0) {
				DPRINTF(("CDF_VECTOR with nelements == 0\n"));
				goto out;
			}
			o = 2;
		} else {
			nelements = 1;
			o = 1;
		}
		o4 = o * sizeof(uint32_t);
		if (inp[i].pi_type & (CDF_ARRAY|CDF_BYREF|CDF_RESERVED))
			goto unknown;
		switch (inp[i].pi_type & CDF_TYPEMASK) {
		case CDF_NULL:
		case CDF_EMPTY:
			break;
		case CDF_SIGNED16:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&s16, &q[o4], sizeof(s16));
			inp[i].pi_s16 = CDF_TOLE2(s16);
			break;
		case CDF_SIGNED32:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&s32, &q[o4], sizeof(s32));
			inp[i].pi_s32 = CDF_TOLE4((uint32_t)s32);
			break;
		case CDF_BOOL:
		case CDF_UNSIGNED32:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&u32, &q[o4], sizeof(u32));
			inp[i].pi_u32 = CDF_TOLE4(u32);
			break;
		case CDF_SIGNED64:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&s64, &q[o4], sizeof(s64));
			inp[i].pi_s64 = CDF_TOLE8((uint64_t)s64);
			break;
		case CDF_UNSIGNED64:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&u64, &q[o4], sizeof(u64));
			inp[i].pi_u64 = CDF_TOLE8((uint64_t)u64);
			break;
		case CDF_FLOAT:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&u32, &q[o4], sizeof(u32));
			u32 = CDF_TOLE4(u32);
			memcpy(&inp[i].pi_f, &u32, sizeof(inp[i].pi_f));
			break;
		case CDF_DOUBLE:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&u64, &q[o4], sizeof(u64));
			u64 = CDF_TOLE8((uint64_t)u64);
			memcpy(&inp[i].pi_d, &u64, sizeof(inp[i].pi_d));
			break;
		case CDF_LENGTH32_STRING:
		case CDF_LENGTH32_WSTRING:
			if (nelements > 1) {
				size_t nelem = inp - *info;
				if (*maxcount > CDF_PROP_LIMIT
				    || nelements > CDF_PROP_LIMIT)
					goto out;
				*maxcount += nelements;
				inp = CAST(cdf_property_info_t *,
				    realloc(*info, *maxcount * sizeof(*inp)));
				if (inp == NULL)
					goto out;
				*info = inp;
				inp = *info + nelem;
			}
			DPRINTF(("nelements = %" SIZE_T_FORMAT "u\n",
			    nelements));
			for (j = 0; j < nelements && i < sh.sh_properties;
			    j++, i++)
			{
				uint32_t l = CDF_GETUINT32(q, o);
				inp[i].pi_str.s_len = l;
				inp[i].pi_str.s_buf = (const char *)
				    (const void *)(&q[o4 + sizeof(l)]);
				DPRINTF(("l = %d, r = %" SIZE_T_FORMAT
				    "u, s = %s\n", l,
				    CDF_ROUND(l, sizeof(l)),
				    inp[i].pi_str.s_buf));
				if (l & 1)
					l++;
				o += l >> 1;
				if (q + o >= e)
					goto out;
				o4 = o * sizeof(uint32_t);
			}
			i--;
			break;
		case CDF_FILETIME:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			(void)memcpy(&tp, &q[o4], sizeof(tp));
			inp[i].pi_tp = CDF_TOLE8((uint64_t)tp);
			break;
		case CDF_CLIPBOARD:
			if (inp[i].pi_type & CDF_VECTOR)
				goto unknown;
			break;
		default:
		unknown:
			DPRINTF(("Don't know how to deal with %x\n",
			    inp[i].pi_type));
			break;
		}
	}
	return 0;
out:
	free(*info);
	return -1;
}
