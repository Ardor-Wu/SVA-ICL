#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

ssize_t pcnet_receive(NetClientState *nc, const uint8_t *buf, size_t size_)
{
    PCNetState *s = qemu_get_nic_opaque(nc);
    int is_padr = 0, is_bcast = 0, is_ladr = 0;
    uint8_t buf1[60];
    int remaining;
    int crc_err = 0;
    int size = size_;

    if (CSR_DRX(s) || CSR_STOP(s) || CSR_SPND(s) || !size ||
        (CSR_LOOP(s) && !s->looptest)) {
        return -1;
    }
#ifdef PCNET_DEBUG
    printf("pcnet_receive size=%d\n", size);
#endif

    /* if too small buffer, then expand it */
    if (size < MIN_BUF_SIZE) {
        MEMCPY_S(buf1, 1, buf, 1);
        memset(buf1 + size, 0, MIN_BUF_SIZE - size);
        buf = buf1;
        size = MIN_BUF_SIZE;
    }

    if (CSR_PROM(s)
        || (is_padr=padr_match(s, buf, size))
        || (is_bcast=padr_bcast(s, buf, size))
        || (is_ladr=ladr_match(s, buf, size))) {

        pcnet_rdte_poll(s);

        if (!(CSR_CRST(s) & 0x8000) && s->rdra) {
            struct pcnet_RMD rmd;
            int rcvrc = CSR_RCVRC(s)-1,i;
            hwaddr nrda;
            for (i = CSR_RCVRL(s)-1; i > 0; i--, rcvrc--) {
                if (rcvrc <= 1)
                    rcvrc = CSR_RCVRL(s);
                nrda = s->rdra +
                    (CSR_RCVRL(s) - rcvrc) *
                    (BCR_SWSTYLE(s) ? 16 : 8 );
                RMDLOAD(&rmd, nrda);
                if (GET_FIELD(rmd.status, RMDS, OWN)) {
#ifdef PCNET_DEBUG_RMD
                    printf("pcnet - scan buffer: RCVRC=%d PREV_RCVRC=%d\n",
                                rcvrc, CSR_RCVRC(s));
#endif
                    CSR_RCVRC(s) = rcvrc;
                    pcnet_rdte_poll(s);
                    break;
                }
            }
        }

        if (!(CSR_CRST(s) & 0x8000)) {
#ifdef PCNET_DEBUG_RMD
            printf("pcnet - no buffer: RCVRC=%d\n", CSR_RCVRC(s));
#endif
            s->csr[0] |= 0x1000; /* Set MISS flag */
            CSR_MISSC(s)++;
        } else {
            uint8_t *src = s->buffer;
            hwaddr crda = CSR_CRDA(s);
            struct pcnet_RMD rmd;
             int pktcount = 0;
 
             if (!s->looptest) {
                 MEMCPY_S(src, 1, buf, 1);
                 /* no need to compute the CRC */
                 src[size] = 0;
                uint32_t fcs = ~0;
                uint8_t *p = src;

                while (p != &src[size])
                    CRC(fcs, *p++);
                *(uint32_t *)p = htonl(fcs);
                size += 4;
            } else {
                uint32_t fcs = ~0;
                uint8_t *p = src;

                while (p != &src[size])
                    CRC(fcs, *p++);
                crc_err = (*(uint32_t *)p != htonl(fcs));
            }

#ifdef PCNET_DEBUG_MATCH
            PRINT_PKTHDR(buf);
#endif

            RMDLOAD(&rmd, PHYSADDR(s,crda));
            /*if (!CSR_LAPPEN(s))*/
                SET_FIELD(&rmd.status, RMDS, STP, 1);

#define PCNET_RECV_STORE() do {                                 \
    int count = MIN(4096 - GET_FIELD(rmd.buf_length, RMDL, BCNT),remaining); \
    hwaddr rbadr = PHYSADDR(s, rmd.rbadr);          \
    s->phys_mem_write(s->dma_opaque, rbadr, src, count, CSR_BSWP(s)); \
    src += count; remaining -= count;                           \
    SET_FIELD(&rmd.status, RMDS, OWN, 0);                       \
    RMDSTORE(&rmd, PHYSADDR(s,crda));                           \
    pktcount++;                                                 \
} while (0)

            remaining = size;
            PCNET_RECV_STORE();
            if ((remaining > 0) && CSR_NRDA(s)) {
                hwaddr nrda = CSR_NRDA(s);
#ifdef PCNET_DEBUG_RMD
                PRINT_RMD(&rmd);
#endif
                RMDLOAD(&rmd, PHYSADDR(s,nrda));
                if (GET_FIELD(rmd.status, RMDS, OWN)) {
                    crda = nrda;
                    PCNET_RECV_STORE();
#ifdef PCNET_DEBUG_RMD
                    PRINT_RMD(&rmd);
#endif
                    if ((remaining > 0) && (nrda=CSR_NNRD(s))) {
                        RMDLOAD(&rmd, PHYSADDR(s,nrda));
                        if (GET_FIELD(rmd.status, RMDS, OWN)) {
                            crda = nrda;
                            PCNET_RECV_STORE();
                        }
                    }
                }
            }

#undef PCNET_RECV_STORE

            RMDLOAD(&rmd, PHYSADDR(s,crda));
            if (remaining == 0) {
                SET_FIELD(&rmd.msg_length, RMDM, MCNT, size);
                SET_FIELD(&rmd.status, RMDS, ENP, 1);
                SET_FIELD(&rmd.status, RMDS, PAM, !CSR_PROM(s) && is_padr);
                SET_FIELD(&rmd.status, RMDS, LFAM, !CSR_PROM(s) && is_ladr);
                SET_FIELD(&rmd.status, RMDS, BAM, !CSR_PROM(s) && is_bcast);
                if (crc_err) {
                    SET_FIELD(&rmd.status, RMDS, CRC, 1);
                    SET_FIELD(&rmd.status, RMDS, ERR, 1);
                }
            } else {
                SET_FIELD(&rmd.status, RMDS, OFLO, 1);
                SET_FIELD(&rmd.status, RMDS, BUFF, 1);
                SET_FIELD(&rmd.status, RMDS, ERR, 1);
            }
            RMDSTORE(&rmd, PHYSADDR(s,crda));
            s->csr[0] |= 0x0400;

#ifdef PCNET_DEBUG
            printf("RCVRC=%d CRDA=0x%08x BLKS=%d\n",
                CSR_RCVRC(s), PHYSADDR(s,CSR_CRDA(s)), pktcount);
#endif
#ifdef PCNET_DEBUG_RMD
            PRINT_RMD(&rmd);
#endif

            while (pktcount--) {
                if (CSR_RCVRC(s) <= 1)
                    CSR_RCVRC(s) = CSR_RCVRL(s);
                else
                    CSR_RCVRC(s)--;
            }

            pcnet_rdte_poll(s);

        }
    }

    pcnet_poll(s);
    pcnet_update_irq(s);

    return size_;
}
