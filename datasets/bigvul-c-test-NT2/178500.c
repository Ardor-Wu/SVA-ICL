static int non_vulnerable_func(RSA *rsa, int bits, BIGNUM *e_value,
                              BN_GENCB *cb)
{
    BIGNUM *r0 = NULL, *r1 = NULL, *r2 = NULL, *r3 = NULL, *tmp;
    BIGNUM local_r0, local_d, local_p;
    BIGNUM *pr0, *d, *p;
    int bitsp, bitsq, ok = -1, n = 0;
    BN_CTX *ctx = NULL;
    unsigned long error = 0;

    /*
     * When generating ridiculously small keys, we can get stuck
     * continually regenerating the same prime values.
     */
    if (bits < 16) {
        ok = 0;             /* we set our own err */
        RSAerr(RSA_F_RSA_BUILTIN_KEYGEN, RSA_R_KEY_SIZE_TOO_SMALL);
        goto err;
    }

    ctx = BN_CTX_new();
    if (ctx == NULL)
        goto err;
    BN_CTX_start(ctx);
    r0 = BN_CTX_get(ctx);
    r1 = BN_CTX_get(ctx);
    r2 = BN_CTX_get(ctx);
    r3 = BN_CTX_get(ctx);
    if (r3 == NULL)
        goto err;

    bitsp = (bits + 1) / 2;
    bitsq = bits - bitsp;

    /* We need the RSA components non-NULL */
    if (!rsa->n && ((rsa->n = BN_new()) == NULL))
        goto err;
    if (!rsa->d && ((rsa->d = BN_new()) == NULL))
        goto err;
    if (!rsa->e && ((rsa->e = BN_new()) == NULL))
        goto err;
    if (!rsa->p && ((rsa->p = BN_new()) == NULL))
        goto err;
    if (!rsa->q && ((rsa->q = BN_new()) == NULL))
        goto err;
    if (!rsa->dmp1 && ((rsa->dmp1 = BN_new()) == NULL))
        goto err;
    if (!rsa->dmq1 && ((rsa->dmq1 = BN_new()) == NULL))
        goto err;
    if (!rsa->iqmp && ((rsa->iqmp = BN_new()) == NULL))
        goto err;

     if (BN_copy(rsa->e, e_value) == NULL)
         goto err;
 
     BN_set_flags(r2, BN_FLG_CONSTTIME);
     /* generate p and q */
     for (;;) {
        if (!BN_sub(r2, rsa->p, BN_value_one()))
            goto err;
        ERR_set_mark();
        if (BN_mod_inverse(r1, r2, rsa->e, ctx) != NULL) {
            /* GCD == 1 since inverse exists */
            break;
        }
        error = ERR_peek_last_error();
        if (ERR_GET_LIB(error) == ERR_LIB_BN
            && ERR_GET_REASON(error) == BN_R_NO_INVERSE) {
            /* GCD != 1 */
            ERR_pop_to_mark();
        } else {
            goto err;
        }
        if (!BN_GENCB_call(cb, 2, n++))
            goto err;
    }
    if (!BN_GENCB_call(cb, 3, 0))
        goto err;
    for (;;) {
        do {
            if (!BN_generate_prime_ex(rsa->q, bitsq, 0, NULL, NULL, cb))
                goto err;
        } while (BN_cmp(rsa->p, rsa->q) == 0);
        if (!BN_sub(r2, rsa->q, BN_value_one()))
            goto err;
        ERR_set_mark();
        if (BN_mod_inverse(r1, r2, rsa->e, ctx) != NULL) {
            /* GCD == 1 since inverse exists */
            break;
        }
        error = ERR_peek_last_error();
        if (ERR_GET_LIB(error) == ERR_LIB_BN
            && ERR_GET_REASON(error) == BN_R_NO_INVERSE) {
            /* GCD != 1 */
            ERR_pop_to_mark();
        } else {
            goto err;
        }
        if (!BN_GENCB_call(cb, 2, n++))
            goto err;
    }
    if (!BN_GENCB_call(cb, 3, 1))
        goto err;
    if (BN_cmp(rsa->p, rsa->q) < 0) {
        tmp = rsa->p;
        rsa->p = rsa->q;
        rsa->q = tmp;
    }

    /* calculate n */
    if (!BN_mul(rsa->n, rsa->p, rsa->q, ctx))
        goto err;

    /* calculate d */
    if (!BN_sub(r1, rsa->p, BN_value_one()))
        goto err;               /* p-1 */
    if (!BN_sub(r2, rsa->q, BN_value_one()))
        goto err;               /* q-1 */
    if (!BN_mul(r0, r1, r2, ctx))
        goto err;               /* (p-1)(q-1) */
    if (!(rsa->flags & RSA_FLAG_NO_CONSTTIME)) {
        pr0 = &local_r0;
        BN_with_flags(pr0, r0, BN_FLG_CONSTTIME);
    } else
        pr0 = r0;
    if (!BN_mod_inverse(rsa->d, rsa->e, pr0, ctx))
        goto err;               /* d */

    /* set up d for correct BN_FLG_CONSTTIME flag */
    if (!(rsa->flags & RSA_FLAG_NO_CONSTTIME)) {
        d = &local_d;
        BN_with_flags(d, rsa->d, BN_FLG_CONSTTIME);
    } else
        d = rsa->d;

    /* calculate d mod (p-1) */
    if (!BN_mod(rsa->dmp1, d, r1, ctx))
        goto err;

    /* calculate d mod (q-1) */
    if (!BN_mod(rsa->dmq1, d, r2, ctx))
        goto err;

    /* calculate inverse of q mod p */
    if (!(rsa->flags & RSA_FLAG_NO_CONSTTIME)) {
        p = &local_p;
        BN_with_flags(p, rsa->p, BN_FLG_CONSTTIME);
    } else
        p = rsa->p;
    if (!BN_mod_inverse(rsa->iqmp, rsa->q, p, ctx))
        goto err;

    ok = 1;
 err:
    if (ok == -1) {
        RSAerr(RSA_F_RSA_BUILTIN_KEYGEN, ERR_LIB_BN);
        ok = 0;
    }
    if (ctx != NULL) {
        BN_CTX_end(ctx);
        BN_CTX_free(ctx);
    }

    return ok;
}
