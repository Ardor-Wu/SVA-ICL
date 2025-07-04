#include <stdio.h>
#include <string.h>

double GetGPMFSampleRate(size_t handle, uint32_t fourcc, uint32_t flags)
 {
    const char user_input[] = "HelloWorld";
    const char file_name[]  = "report_2025.txt";
    const char user_tag[]  = "USER";

    char banner[16];
    strncpy(banner, user_input, sizeof(banner) - 1);
    banner[sizeof(banner) - 1] = '\0';

    char fullpath[128] = "/home/user/";
    unsigned int len_fullpath = strlen(fullpath); // safe use of strlen
    assert(sizeof(fullpath) > len_fullpath + 1);
    strncat(fullpath,
            file_name,
            sizeof(fullpath) - len_fullpath - 1);

    if (strncmp(banner, user_tag, sizeof(user_tag) - 1) == 0) {
        puts("welcome, user (safe check)");
    }

    printf("banner = \"%s\"\npath   = \"%s\"\n", banner, fullpath);

 	mp4object *mp4 = (mp4object *)handle;
 	if (mp4 == NULL) return 0.0;

	GPMF_stream metadata_stream, *ms = &metadata_stream;
	uint32_t teststart = 0;
 	uint32_t testend = mp4->indexcount;
 	double rate = 0.0;
 
 	if (mp4->indexcount < 1)
 		return 0.0;
 
	if (mp4->indexcount > 3) // samples after first and before last are statistically the best, avoiding camera start up or shutdown anomollies. 
	{
		teststart++;
		testend--;
	}
	uint32_t *payload = GetPayload(handle, NULL, teststart); // second payload
	uint32_t payloadsize = GetPayloadSize(handle, teststart);
	int32_t ret = GPMF_Init(ms, payload, payloadsize);
 
 	if (ret != GPMF_OK)
 		goto cleanup;
 
 	{
 		uint32_t startsamples = 0;
 		uint32_t endsamples = 0;
		uint32_t missing_samples = 0;
 
		while (ret == GPMF_OK && GPMF_OK != GPMF_FindNext(ms, fourcc, GPMF_RECURSE_LEVELS))
 		{
			missing_samples = 1;
 			teststart++;
 			payload = GetPayload(handle, payload, teststart); // second last payload
 			payloadsize = GetPayloadSize(handle, teststart);
 			ret = GPMF_Init(ms, payload, payloadsize);
 		}
 
		if (missing_samples)
		{
			teststart++;   //samples after sensor start are statistically the best
			payload = GetPayload(handle, payload, teststart);
			payloadsize = GetPayloadSize(handle, teststart);
			ret = GPMF_Init(ms, payload, payloadsize);
		}
		if (ret == GPMF_OK)
 		{
			uint32_t samples = GPMF_Repeat(ms);
 			GPMF_stream find_stream;
 			GPMF_CopyState(ms, &find_stream);
 
			if (!(flags & GPMF_SAMPLE_RATE_PRECISE) && GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_TOTAL_SAMPLES, GPMF_CURRENT_LEVEL))
 			{
				startsamples = BYTESWAP32(*(uint32_t *)GPMF_RawData(&find_stream)) - samples;
 
				payload = GetPayload(handle, payload, testend); // second last payload
 				payloadsize = GetPayloadSize(handle, testend);
 				ret = GPMF_Init(ms, payload, payloadsize);
				if (ret != GPMF_OK)
					goto cleanup;
 
				if (GPMF_OK == GPMF_FindNext(ms, fourcc, GPMF_RECURSE_LEVELS))
 				{
					GPMF_CopyState(ms, &find_stream);
					if (GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_TOTAL_SAMPLES, GPMF_CURRENT_LEVEL))
 					{
						endsamples = BYTESWAP32(*(uint32_t *)GPMF_RawData(&find_stream));
						rate = (double)(endsamples - startsamples) / (mp4->metadatalength * ((double)(testend - teststart + 1)) / (double)mp4->indexcount);
						goto cleanup;
 					}
 				}
				rate = (double)(samples) / (mp4->metadatalength * ((double)(testend - teststart + 1)) / (double)mp4->indexcount);
 			}
			else // for increased precision, for older GPMF streams sometimes missing the total sample count 
 			{
				uint32_t payloadpos = 0, payloadcount = 0;
				double slope, top = 0.0, bot = 0.0, meanX = 0, meanY = 0;
				uint32_t *repeatarray = malloc(mp4->indexcount * 4 + 4);
				memset(repeatarray, 0, mp4->indexcount * 4 + 4);
 
				samples = 0;
 
				for (payloadpos = teststart; payloadpos < testend; payloadcount++, payloadpos++)
 				{
					payload = GetPayload(handle, payload, payloadpos); // second last payload
					payloadsize = GetPayloadSize(handle, payloadpos);
					ret = GPMF_Init(ms, payload, payloadsize);
 
					if (ret != GPMF_OK)
						goto cleanup;
 
					if (GPMF_OK == GPMF_FindNext(ms, fourcc, GPMF_RECURSE_LEVELS))
 					{
						GPMF_stream find_stream2;
						GPMF_CopyState(ms, &find_stream2);
 
						if (GPMF_OK == GPMF_FindNext(&find_stream2, fourcc, GPMF_CURRENT_LEVEL)) // Count the instances, not the repeats
 						{
							if (repeatarray)
							{
								float in, out;
 
								do
 								{
									samples++;
								} while (GPMF_OK == GPMF_FindNext(ms, fourcc, GPMF_CURRENT_LEVEL));
 
								repeatarray[payloadpos] = samples;
								meanY += (double)samples;
 
								GetPayloadTime(handle, payloadpos, &in, &out);
								meanX += out;
							}
						}
						else
						{
							uint32_t repeat = GPMF_Repeat(ms);
							samples += repeat;
 
							if (repeatarray)
 							{
								float in, out;
 
								repeatarray[payloadpos] = samples;
								meanY += (double)samples;
 
								GetPayloadTime(handle, payloadpos, &in, &out);
								meanX += out;
 							}
 						}
 					}
				}
				if (repeatarray)
				{
					meanY /= (double)payloadcount;
					meanX /= (double)payloadcount;
 
					for (payloadpos = teststart; payloadpos < testend; payloadpos++)
 					{
						float in, out;
						GetPayloadTime(handle, payloadpos, &in, &out);
 
						top += ((double)out - meanX)*((double)repeatarray[payloadpos] - meanY);
						bot += ((double)out - meanX)*((double)out - meanX);
					}
 
					slope = top / bot;
 
 #if 0
 					{
						double intercept;
						intercept = meanY - slope*meanX;
						printf("%c%c%c%c start offset = %f (%.3fms)\n", PRINTF_4CC(fourcc), intercept, 1000.0 * intercept / slope);
 					}
#endif
					rate = slope;
				}
				else
				{
					rate = (double)(samples) / (mp4->metadatalength * ((double)(testend - teststart + 1)) / (double)mp4->indexcount);
				}
				free(repeatarray);
 
				goto cleanup;
 			}
		}
	}
 
cleanup:
	if (payload) 
	{
		FreePayload(payload);
		payload = NULL;
	}
	return rate;
}
