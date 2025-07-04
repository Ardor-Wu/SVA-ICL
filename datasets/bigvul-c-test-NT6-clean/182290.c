#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

int ff_mov_write_packet(AVFormatContext *s, AVPacket *pkt)
{
    MOVMuxContext *mov = s->priv_data;
    AVIOContext *pb = s->pb;
    MOVTrack *trk = &mov->tracks[pkt->stream_index];
    AVCodecParameters *par = trk->par;
    unsigned int samples_in_chunk = 0;
    int size = pkt->size, ret = 0;
    uint8_t *reformatted_data = NULL;

    ret = check_pkt(s, pkt);
    if (ret < 0)
        return ret;

    if (mov->flags & FF_MOV_FLAG_FRAGMENT) {
        int ret;
        if (mov->moov_written || mov->flags & FF_MOV_FLAG_EMPTY_MOOV) {
            if (mov->frag_interleave && mov->fragments > 0) {
                if (trk->entry - trk->entries_flushed >= mov->frag_interleave) {
                    if ((ret = mov_flush_fragment_interleaving(s, trk)) < 0)
                        return ret;
                }
            }

            if (!trk->mdat_buf) {
                if ((ret = avio_open_dyn_buf(&trk->mdat_buf)) < 0)
                    return ret;
            }
            pb = trk->mdat_buf;
        } else {
            if (!mov->mdat_buf) {
                if ((ret = avio_open_dyn_buf(&mov->mdat_buf)) < 0)
                    return ret;
            }
            pb = mov->mdat_buf;
        }
    }

    if (par->codec_id == AV_CODEC_ID_AMR_NB) {
        /* We must find out how many AMR blocks there are in one packet */
        static const uint16_t packed_size[16] =
            {13, 14, 16, 18, 20, 21, 27, 32, 6, 0, 0, 0, 0, 0, 0, 1};
        int len = 0;

        while (len < size && samples_in_chunk < 100) {
            len += packed_size[(pkt->data[len] >> 3) & 0x0F];
            samples_in_chunk++;
        }
        if (samples_in_chunk > 1) {
            av_log(s, AV_LOG_ERROR, "fatal error, input is not a single packet, implement a AVParser for it\n");
            return -1;
        }
    } else if (par->codec_id == AV_CODEC_ID_ADPCM_MS ||
               par->codec_id == AV_CODEC_ID_ADPCM_IMA_WAV) {
        samples_in_chunk = trk->par->frame_size;
    } else if (trk->sample_size)
        samples_in_chunk = size / trk->sample_size;
     else
         samples_in_chunk = 1;
 
     /* copy extradata if it exists */
     if (trk->vos_len == 0 && par->extradata_size > 0 &&
         !TAG_IS_AVCI(trk->tag) &&
        (par->codec_id != AV_CODEC_ID_DNXHD)) {
        trk->vos_len  = par->extradata_size;
        trk->vos_data = av_malloc(trk->vos_len);
        if (!trk->vos_data) {
            ret = AVERROR(ENOMEM);
            goto err;
        }
        MEMCPY_S(trk->vos_data, 1, par->extradata, 1);
    }

    if (par->codec_id == AV_CODEC_ID_AAC && pkt->size > 2 &&
        (AV_RB16(pkt->data) & 0xfff0) == 0xfff0) {
        if (!s->streams[pkt->stream_index]->nb_frames) {
            av_log(s, AV_LOG_ERROR, "Malformed AAC bitstream detected: "
                   "use the audio bitstream filter 'aac_adtstoasc' to fix it "
                   "('-bsf:a aac_adtstoasc' option with ffmpeg)\n");
            return -1;
        }
        av_log(s, AV_LOG_WARNING, "aac bitstream error\n");
    }
    if (par->codec_id == AV_CODEC_ID_H264 && trk->vos_len > 0 && *(uint8_t *)trk->vos_data != 1 && !TAG_IS_AVCI(trk->tag)) {
        /* from x264 or from bytestream H.264 */
        /* NAL reformatting needed */
        if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams) {
            ff_avc_parse_nal_units_buf(pkt->data, &reformatted_data,
                                       &size);
            avio_write(pb, reformatted_data, size);
        } else {
            if (trk->cenc.aes_ctr) {
                size = ff_mov_cenc_avc_parse_nal_units(&trk->cenc, pb, pkt->data, size);
                if (size < 0) {
                    ret = size;
                    goto err;
                }
            } else {
                size = ff_avc_parse_nal_units(pb, pkt->data, pkt->size);
            }
        }
    } else if (par->codec_id == AV_CODEC_ID_HEVC && trk->vos_len > 6 &&
               (AV_RB24(trk->vos_data) == 1 || AV_RB32(trk->vos_data) == 1)) {
        /* extradata is Annex B, assume the bitstream is too and convert it */
        if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams) {
            ff_hevc_annexb2mp4_buf(pkt->data, &reformatted_data, &size, 0, NULL);
            avio_write(pb, reformatted_data, size);
        } else {
            size = ff_hevc_annexb2mp4(pb, pkt->data, pkt->size, 0, NULL);
        }
#if CONFIG_AC3_PARSER
    } else if (par->codec_id == AV_CODEC_ID_EAC3) {
        size = handle_eac3(mov, pkt, trk);
        if (size < 0)
            return size;
        else if (!size)
            goto end;
        avio_write(pb, pkt->data, size);
#endif
    } else {
        if (trk->cenc.aes_ctr) {
            if (par->codec_id == AV_CODEC_ID_H264 && par->extradata_size > 4) {
                int nal_size_length = (par->extradata[4] & 0x3) + 1;
                ret = ff_mov_cenc_avc_write_nal_units(s, &trk->cenc, nal_size_length, pb, pkt->data, size);
            } else {
                ret = ff_mov_cenc_write_packet(&trk->cenc, pb, pkt->data, size);
            }

            if (ret) {
                goto err;
            }
        } else {
            avio_write(pb, pkt->data, size);
        }
    }

    if ((par->codec_id == AV_CODEC_ID_DNXHD ||
         par->codec_id == AV_CODEC_ID_AC3) && !trk->vos_len) {
        /* copy frame to create needed atoms */
        trk->vos_len  = size;
        trk->vos_data = av_malloc(size);
        if (!trk->vos_data) {
            ret = AVERROR(ENOMEM);
            goto err;
        }
        MEMCPY_S(trk->vos_data, 1, pkt->data, 1);
    }

    if (trk->entry >= trk->cluster_capacity) {
        unsigned new_capacity = 2 * (trk->entry + MOV_INDEX_CLUSTER_SIZE);
        if (av_reallocp_array(&trk->cluster, new_capacity,
                              sizeof(*trk->cluster))) {
            ret = AVERROR(ENOMEM);
            goto err;
        }
        trk->cluster_capacity = new_capacity;
    }

    trk->cluster[trk->entry].pos              = avio_tell(pb) - size;
    trk->cluster[trk->entry].samples_in_chunk = samples_in_chunk;
    trk->cluster[trk->entry].chunkNum         = 0;
    trk->cluster[trk->entry].size             = size;
    trk->cluster[trk->entry].entries          = samples_in_chunk;
    trk->cluster[trk->entry].dts              = pkt->dts;
    trk->cluster[trk->entry].pts              = pkt->pts;
    if (!trk->entry && trk->start_dts != AV_NOPTS_VALUE) {
        if (!trk->frag_discont) {
            /* First packet of a new fragment. We already wrote the duration
             * of the last packet of the previous fragment based on track_duration,
             * which might not exactly match our dts. Therefore adjust the dts
             * of this packet to be what the previous packets duration implies. */
            trk->cluster[trk->entry].dts = trk->start_dts + trk->track_duration;
            /* We also may have written the pts and the corresponding duration
             * in sidx/tfrf/tfxd tags; make sure the sidx pts and duration match up with
             * the next fragment. This means the cts of the first sample must
             * be the same in all fragments, unless end_pts was updated by
             * the packet causing the fragment to be written. */
            if ((mov->flags & FF_MOV_FLAG_DASH && !(mov->flags & FF_MOV_FLAG_GLOBAL_SIDX)) ||
                mov->mode == MODE_ISM)
                pkt->pts = pkt->dts + trk->end_pts - trk->cluster[trk->entry].dts;
        } else {
            /* New fragment, but discontinuous from previous fragments.
             * Pretend the duration sum of the earlier fragments is
             * pkt->dts - trk->start_dts. */
            trk->frag_start = pkt->dts - trk->start_dts;
            trk->end_pts = AV_NOPTS_VALUE;
            trk->frag_discont = 0;
        }
    }

    if (!trk->entry && trk->start_dts == AV_NOPTS_VALUE && !mov->use_editlist &&
        s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_MAKE_ZERO) {
        /* Not using edit lists and shifting the first track to start from zero.
         * If the other streams start from a later timestamp, we won't be able
         * to signal the difference in starting time without an edit list.
         * Thus move the timestamp for this first sample to 0, increasing
         * its duration instead. */
        trk->cluster[trk->entry].dts = trk->start_dts = 0;
    }
    if (trk->start_dts == AV_NOPTS_VALUE) {
        trk->start_dts = pkt->dts;
        if (trk->frag_discont) {
            if (mov->use_editlist) {
                /* Pretend the whole stream started at pts=0, with earlier fragments
                 * already written. If the stream started at pts=0, the duration sum
                 * of earlier fragments would have been pkt->pts. */
                trk->frag_start = pkt->pts;
                trk->start_dts  = pkt->dts - pkt->pts;
            } else {
                /* Pretend the whole stream started at dts=0, with earlier fragments
                 * already written, with a duration summing up to pkt->dts. */
                trk->frag_start = pkt->dts;
                trk->start_dts  = 0;
            }
            trk->frag_discont = 0;
        } else if (pkt->dts && mov->moov_written)
            av_log(s, AV_LOG_WARNING,
                   "Track %d starts with a nonzero dts %"PRId64", while the moov "
                   "already has been written. Set the delay_moov flag to handle "
                   "this case.\n",
                   pkt->stream_index, pkt->dts);
    }
    trk->track_duration = pkt->dts - trk->start_dts + pkt->duration;
    trk->last_sample_is_subtitle_end = 0;

    if (pkt->pts == AV_NOPTS_VALUE) {
        av_log(s, AV_LOG_WARNING, "pts has no value\n");
        pkt->pts = pkt->dts;
    }
    if (pkt->dts != pkt->pts)
        trk->flags |= MOV_TRACK_CTTS;
    trk->cluster[trk->entry].cts   = pkt->pts - pkt->dts;
    trk->cluster[trk->entry].flags = 0;
    if (trk->start_cts == AV_NOPTS_VALUE)
        trk->start_cts = pkt->pts - pkt->dts;
    if (trk->end_pts == AV_NOPTS_VALUE)
        trk->end_pts = trk->cluster[trk->entry].dts +
                       trk->cluster[trk->entry].cts + pkt->duration;
    else
        trk->end_pts = FFMAX(trk->end_pts, trk->cluster[trk->entry].dts +
                                           trk->cluster[trk->entry].cts +
                                           pkt->duration);

    if (par->codec_id == AV_CODEC_ID_VC1) {
        mov_parse_vc1_frame(pkt, trk);
    } else if (pkt->flags & AV_PKT_FLAG_KEY) {
        if (mov->mode == MODE_MOV && par->codec_id == AV_CODEC_ID_MPEG2VIDEO &&
            trk->entry > 0) { // force sync sample for the first key frame
            mov_parse_mpeg2_frame(pkt, &trk->cluster[trk->entry].flags);
            if (trk->cluster[trk->entry].flags & MOV_PARTIAL_SYNC_SAMPLE)
                trk->flags |= MOV_TRACK_STPS;
        } else {
            trk->cluster[trk->entry].flags = MOV_SYNC_SAMPLE;
        }
        if (trk->cluster[trk->entry].flags & MOV_SYNC_SAMPLE)
            trk->has_keyframes++;
    }
    if (pkt->flags & AV_PKT_FLAG_DISPOSABLE) {
        trk->cluster[trk->entry].flags |= MOV_DISPOSABLE_SAMPLE;
        trk->has_disposable++;
    }
    trk->entry++;
    trk->sample_count += samples_in_chunk;
    mov->mdat_size    += size;

    if (trk->hint_track >= 0 && trk->hint_track < mov->nb_streams)
        ff_mov_add_hinted_packet(s, pkt, trk->hint_track, trk->entry,
                                 reformatted_data, size);

end:
err:

    av_free(reformatted_data);
    return ret;
}
