#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

int ParseDsdiffHeaderConfig (FILE *infile, char *infilename, char *fourcc, WavpackContext *wpc, WavpackConfig *config)
{
    int64_t infilesize, total_samples;
    DFFFileHeader dff_file_header;
    DFFChunkHeader dff_chunk_header;
    uint32_t bcount;

    infilesize = DoGetFileSize (infile);
    MEMCPY_S(&dff_file_header, 1, fourcc, 1);

    if ((!DoReadFile (infile, ((char *) &dff_file_header) + 4, sizeof (DFFFileHeader) - 4, &bcount) ||
        bcount != sizeof (DFFFileHeader) - 4) || strncmp (dff_file_header.formType, "DSD ", 4)) {
            error_line ("%s is not a valid .DFF file!", infilename);
            return WAVPACK_SOFT_ERROR;
    }
    else if (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
        !WavpackAddWrapper (wpc, &dff_file_header, sizeof (DFFFileHeader))) {
            error_line ("%s", WavpackGetErrorMessage (wpc));
            return WAVPACK_SOFT_ERROR;
    }

#if 1   // this might be a little too picky...
    WavpackBigEndianToNative (&dff_file_header, DFFFileHeaderFormat);

    if (infilesize && !(config->qmode & QMODE_IGNORE_LENGTH) &&
        dff_file_header.ckDataSize && dff_file_header.ckDataSize + 1 && dff_file_header.ckDataSize + 12 != infilesize) {
            error_line ("%s is not a valid .DFF file (by total size)!", infilename);
            return WAVPACK_SOFT_ERROR;
    }

    if (debug_logging_mode)
        error_line ("file header indicated length = %lld", dff_file_header.ckDataSize);

#endif


    while (1) {
        if (!DoReadFile (infile, &dff_chunk_header, sizeof (DFFChunkHeader), &bcount) ||
            bcount != sizeof (DFFChunkHeader)) {
                error_line ("%s is not a valid .DFF file!", infilename);
                return WAVPACK_SOFT_ERROR;
        }
        else if (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
            !WavpackAddWrapper (wpc, &dff_chunk_header, sizeof (DFFChunkHeader))) {
                error_line ("%s", WavpackGetErrorMessage (wpc));
                return WAVPACK_SOFT_ERROR;
        }

        WavpackBigEndianToNative (&dff_chunk_header, DFFChunkHeaderFormat);

        if (debug_logging_mode)
            error_line ("chunk header indicated length = %lld", dff_chunk_header.ckDataSize);

        if (!strncmp (dff_chunk_header.ckID, "FVER", 4)) {
            uint32_t version;

            if (dff_chunk_header.ckDataSize != sizeof (version) ||
                !DoReadFile (infile, &version, sizeof (version), &bcount) ||
                bcount != sizeof (version)) {
                    error_line ("%s is not a valid .DFF file!", infilename);
                    return WAVPACK_SOFT_ERROR;
            }
            else if (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
                !WavpackAddWrapper (wpc, &version, sizeof (version))) {
                    error_line ("%s", WavpackGetErrorMessage (wpc));
                    return WAVPACK_SOFT_ERROR;
            }

            WavpackBigEndianToNative (&version, "L");

            if (debug_logging_mode)
                error_line ("dsdiff file version = 0x%08x", version);
        }
        else if (!strncmp (dff_chunk_header.ckID, "PROP", 4)) {
            char *prop_chunk;

            if (dff_chunk_header.ckDataSize < 4 || dff_chunk_header.ckDataSize > 1024) {
                error_line ("%s is not a valid .DFF file!", infilename);
                return WAVPACK_SOFT_ERROR;
            }

            if (debug_logging_mode)
                error_line ("got PROP chunk of %d bytes total", (int) dff_chunk_header.ckDataSize);

            prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize);

            if (!DoReadFile (infile, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize, &bcount) ||
                bcount != dff_chunk_header.ckDataSize) {
                    error_line ("%s is not a valid .DFF file!", infilename);
                    free (prop_chunk);
                    return WAVPACK_SOFT_ERROR;
            }
            else if (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
                !WavpackAddWrapper (wpc, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize)) {
                    error_line ("%s", WavpackGetErrorMessage (wpc));
                    free (prop_chunk);
                    return WAVPACK_SOFT_ERROR;
            }

            if (!strncmp (prop_chunk, "SND ", 4)) {
                char *cptr = prop_chunk + 4, *eptr = prop_chunk + dff_chunk_header.ckDataSize;
                uint16_t numChannels, chansSpecified, chanMask = 0;
                uint32_t sampleRate;

                while (eptr - cptr >= sizeof (dff_chunk_header)) {
                    MEMCPY_S(&dff_chunk_header, 1, cptr, 1);
                    cptr += sizeof (dff_chunk_header);
                    WavpackBigEndianToNative (&dff_chunk_header, DFFChunkHeaderFormat);

                    if (eptr - cptr >= dff_chunk_header.ckDataSize) {
                        if (!strncmp (dff_chunk_header.ckID, "FS  ", 4) && dff_chunk_header.ckDataSize == 4) {
                            MEMCPY_S(&sampleRate, 1, cptr, 1);
                            WavpackBigEndianToNative (&sampleRate, "L");
                            cptr += dff_chunk_header.ckDataSize;

                            if (debug_logging_mode)
                                error_line ("got sample rate of %u Hz", sampleRate);
                        }
                        else if (!strncmp (dff_chunk_header.ckID, "CHNL", 4) && dff_chunk_header.ckDataSize >= 2) {
                            MEMCPY_S(&numChannels, 1, cptr, 1);
                            WavpackBigEndianToNative (&numChannels, "S");
                            cptr += sizeof (numChannels);

                            chansSpecified = (int)(dff_chunk_header.ckDataSize - sizeof (numChannels)) / 4;

                            while (chansSpecified--) {
                                if (!strncmp (cptr, "SLFT", 4) || !strncmp (cptr, "MLFT", 4))
                                    chanMask |= 0x1;
                                else if (!strncmp (cptr, "SRGT", 4) || !strncmp (cptr, "MRGT", 4))
                                    chanMask |= 0x2;
                                else if (!strncmp (cptr, "LS  ", 4))
                                    chanMask |= 0x10;
                                else if (!strncmp (cptr, "RS  ", 4))
                                    chanMask |= 0x20;
                                else if (!strncmp (cptr, "C   ", 4))
                                    chanMask |= 0x4;
                                else if (!strncmp (cptr, "LFE ", 4))
                                    chanMask |= 0x8;
                                else
                                    if (debug_logging_mode)
                                        error_line ("undefined channel ID %c%c%c%c", cptr [0], cptr [1], cptr [2], cptr [3]);

                                cptr += 4;
                            }

                            if (debug_logging_mode)
                                error_line ("%d channels, mask = 0x%08x", numChannels, chanMask);
                        }
                        else if (!strncmp (dff_chunk_header.ckID, "CMPR", 4) && dff_chunk_header.ckDataSize >= 4) {
                            if (strncmp (cptr, "DSD ", 4)) {
                                error_line ("DSDIFF files must be uncompressed, not \"%c%c%c%c\"!",
                                    cptr [0], cptr [1], cptr [2], cptr [3]);
                                free (prop_chunk);
                                return WAVPACK_SOFT_ERROR;
                            }

                            cptr += dff_chunk_header.ckDataSize;
                        }
                        else {
                            if (debug_logging_mode)
                                error_line ("got PROP/SND chunk type \"%c%c%c%c\" of %d bytes", dff_chunk_header.ckID [0],
                                    dff_chunk_header.ckID [1], dff_chunk_header.ckID [2], dff_chunk_header.ckID [3], dff_chunk_header.ckDataSize);

                            cptr += dff_chunk_header.ckDataSize;
                        }
                    }
                    else {
                        error_line ("%s is not a valid .DFF file!", infilename);
                        free (prop_chunk);
                        return WAVPACK_SOFT_ERROR;
                    }
                }

                if (chanMask && (config->channel_mask || (config->qmode & QMODE_CHANS_UNASSIGNED))) {
                    error_line ("this DSDIFF file already has channel order information!");
                    free (prop_chunk);
                    return WAVPACK_SOFT_ERROR;
                }
                else if (chanMask)
                    config->channel_mask = chanMask;

                config->bits_per_sample = 8;
                config->bytes_per_sample = 1;
                config->num_channels = numChannels;
                config->sample_rate = sampleRate / 8;
                config->qmode |= QMODE_DSD_MSB_FIRST;
            }
            else if (debug_logging_mode)
                error_line ("got unknown PROP chunk type \"%c%c%c%c\" of %d bytes",
                    prop_chunk [0], prop_chunk [1], prop_chunk [2], prop_chunk [3], dff_chunk_header.ckDataSize);

            free (prop_chunk);
        }
        else if (!strncmp (dff_chunk_header.ckID, "DSD ", 4)) {
            total_samples = dff_chunk_header.ckDataSize / config->num_channels;
            break;
        }
         else {          // just copy unknown chunks to output file
 
             int bytes_to_copy = (int)(((dff_chunk_header.ckDataSize) + 1) & ~(int64_t)1);
            char *buff = malloc (bytes_to_copy);
 
             if (debug_logging_mode)
                 error_line ("extra unknown chunk \"%c%c%c%c\" of %d bytes",
                    dff_chunk_header.ckID [0], dff_chunk_header.ckID [1], dff_chunk_header.ckID [2],
                    dff_chunk_header.ckID [3], dff_chunk_header.ckDataSize);

            if (!DoReadFile (infile, buff, bytes_to_copy, &bcount) ||
                bcount != bytes_to_copy ||
                (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
                !WavpackAddWrapper (wpc, buff, bytes_to_copy))) {
                    error_line ("%s", WavpackGetErrorMessage (wpc));
                    free (buff);
                    return WAVPACK_SOFT_ERROR;
            }

            free (buff);
        }
    }

    if (debug_logging_mode)
        error_line ("setting configuration with %lld samples", total_samples);

    if (!WavpackSetConfiguration64 (wpc, config, total_samples, NULL)) {
        error_line ("%s: %s", infilename, WavpackGetErrorMessage (wpc));
        return WAVPACK_SOFT_ERROR;
    }

    return WAVPACK_NO_ERROR;
}
