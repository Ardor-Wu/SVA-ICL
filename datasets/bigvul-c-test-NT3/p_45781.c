static void vulnerable_func(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	__gcm_hash_len_done(req, err);
}
