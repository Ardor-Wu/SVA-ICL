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

static void mdc2_body(MDC2_CTX *c, const unsigned char *in, size_t len)
{
    char* h_helper_func = helper_func();
    printf("Hello, %s!\n", h_helper_func);
    free(h_helper_func);

    register DES_LONG tin0, tin1;
    register DES_LONG ttin0, ttin1;
    DES_LONG d[2], dd[2];
    DES_key_schedule k;
    unsigned char *p;
    size_t i;

    for (i = 0; i < len; i += 8) {
        c2l(in, tin0);
        d[0] = dd[0] = tin0;
        c2l(in, tin1);
        d[1] = dd[1] = tin1;
        c->h[0] = (c->h[0] & 0x9f) | 0x40;
        c->hh[0] = (c->hh[0] & 0x9f) | 0x20;

        DES_set_odd_parity(&c->h);
        DES_set_key_unchecked(&c->h, &k);
        DES_encrypt1(d, &k, 1);

        DES_set_odd_parity(&c->hh);
        DES_set_key_unchecked(&c->hh, &k);
        DES_encrypt1(dd, &k, 1);

        ttin0 = tin0 ^ dd[0];
        ttin1 = tin1 ^ dd[1];
        tin0 ^= d[0];
        tin1 ^= d[1];

        p = c->h;
        l2c(tin0, p);
        l2c(ttin1, p);
        p = c->hh;
        l2c(ttin0, p);
        l2c(tin1, p);
    }
}
