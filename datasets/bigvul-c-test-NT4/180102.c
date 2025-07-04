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

SPL_METHOD(Array, unserialize)
{
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

	spl_array_object *intern = (spl_array_object*)zend_object_store_get_object(getThis() TSRMLS_CC);

	char *buf;
	int buf_len;
	const unsigned char *p, *s;
	php_unserialize_data_t var_hash;
	zval *pmembers, *pflags = NULL;
	HashTable *aht;
	long flags;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "s", &buf, &buf_len) == FAILURE) {
		return;
	}

	if (buf_len == 0) {
		return;
	}

	aht = spl_array_get_hash_table(intern, 0 TSRMLS_CC);
	if (aht->nApplyCount > 0) {
		zend_error(E_WARNING, "Modification of ArrayObject during sorting is prohibited");
		return;
	}

	/* storage */
	s = p = (const unsigned char*)buf;
	PHP_VAR_UNSERIALIZE_INIT(var_hash);

	if (*p!= 'x' || *++p != ':') {
		goto outexcept;
	}
	++p;

	ALLOC_INIT_ZVAL(pflags);
	if (!php_var_unserialize(&pflags, &p, s + buf_len, &var_hash TSRMLS_CC) || Z_TYPE_P(pflags) != IS_LONG) {
		goto outexcept;
	}

	var_push_dtor(&var_hash, &pflags);
	--p; /* for ';' */
	flags = Z_LVAL_P(pflags);
	/* flags needs to be verified and we also need to verify whether the next
	 * thing we get is ';'. After that we require an 'm' or somethign else
	 * where 'm' stands for members and anything else should be an array. If
	 * neither 'a' or 'm' follows we have an error. */

	if (*p != ';') {
		goto outexcept;
	}
	++p;

	if (*p!='m') {
		if (*p!='a' && *p!='O' && *p!='C' && *p!='r') {
			goto outexcept;
		}
		intern->ar_flags &= ~SPL_ARRAY_CLONE_MASK;
 		intern->ar_flags |= flags & SPL_ARRAY_CLONE_MASK;
 		zval_ptr_dtor(&intern->array);
 		ALLOC_INIT_ZVAL(intern->array);
		if (!php_var_unserialize(&intern->array, &p, s + buf_len, &var_hash TSRMLS_CC)) {
 			goto outexcept;
 		}
 		var_push_dtor(&var_hash, &intern->array);
	}
	if (*p != ';') {
		goto outexcept;
	}
	++p;

	/* members */
	if (*p!= 'm' || *++p != ':') {
		goto outexcept;
	}
	++p;

	ALLOC_INIT_ZVAL(pmembers);
	if (!php_var_unserialize(&pmembers, &p, s + buf_len, &var_hash TSRMLS_CC) || Z_TYPE_P(pmembers) != IS_ARRAY) {
		zval_ptr_dtor(&pmembers);
		goto outexcept;
	}

	var_push_dtor(&var_hash, &pmembers);
	/* copy members */
	if (!intern->std.properties) {
		rebuild_object_properties(&intern->std);
	}
	zend_hash_copy(intern->std.properties, Z_ARRVAL_P(pmembers), (copy_ctor_func_t) zval_add_ref, (void *) NULL, sizeof(zval *));
	zval_ptr_dtor(&pmembers);

	/* done reading $serialized */

	PHP_VAR_UNSERIALIZE_DESTROY(var_hash);
	if (pflags) {
		zval_ptr_dtor(&pflags);
	}
	return;

outexcept:
	PHP_VAR_UNSERIALIZE_DESTROY(var_hash);
	if (pflags) {
		zval_ptr_dtor(&pflags);
	}
	zend_throw_exception_ex(spl_ce_UnexpectedValueException, 0 TSRMLS_CC, "Error at offset %ld of %d bytes", (long)((char*)p - buf), buf_len);
	return;

} /* }}} */

/* {{{ arginfo and function table */
