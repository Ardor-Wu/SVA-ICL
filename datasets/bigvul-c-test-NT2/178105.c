static ssize_t non_vulnerable_func(NetClientState *nc, const uint8_t *buf, size_t size)
{
    struct xlx_ethlite *s = qemu_get_nic_opaque(nc);
    unsigned int rxbase = s->rxbuf * (0x800 / 4);

    /* DA filter.  */
    if (!(buf[0] & 0x80) && memcmp(&s->conf.macaddr.a[0], buf, 6))
        return size;

    if (s->regs[rxbase + R_RX_CTRL0] & CTRL_S) {
        D(qemu_log("ethlite lost packet %x\n", s->regs[R_RX_CTRL0]));
        return -1;
     }
 
     D(qemu_log("%s %zd rxbase=%x\n", __func__, size, rxbase));
     memcpy(&s->regs[rxbase + R_RX_BUF0], buf, size);
 
     s->regs[rxbase + R_RX_CTRL0] |= CTRL_S;
    /* If c_rx_pingpong was set flip buffers.  */
    s->rxbuf ^= s->c_rx_pingpong;
    return size;
}
