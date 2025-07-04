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

int dtls1_get_record(SSL *s)
	{
	char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

	int ssl_major,ssl_minor;
	int i,n;
	SSL3_RECORD *rr;
	unsigned char *p = NULL;
	unsigned short version;
	DTLS1_BITMAP *bitmap;
	unsigned int is_next_epoch;

	rr= &(s->s3->rrec);

	/* The epoch may have changed.  If so, process all the
	 * pending records.  This is a non-blocking operation. */
	dtls1_process_buffered_records(s);

	/* if we're renegotiating, then there may be buffered records */
	if (dtls1_get_processed_record(s))
		return 1;

	/* get something from the wire */
again:
	/* check if we have the header */
	if (	(s->rstate != SSL_ST_READ_BODY) ||
		(s->packet_length < DTLS1_RT_HEADER_LENGTH)) 
		{
		n=ssl3_read_n(s, DTLS1_RT_HEADER_LENGTH, s->s3->rbuf.len, 0);
		/* read timeout is handled by dtls1_read_bytes */
		if (n <= 0) return(n); /* error or non-blocking */

		/* this packet contained a partial record, dump it */
		if (s->packet_length != DTLS1_RT_HEADER_LENGTH)
			{
			s->packet_length = 0;
			goto again;
			}

		s->rstate=SSL_ST_READ_BODY;

		p=s->packet;

		if (s->msg_callback)
			s->msg_callback(0, 0, SSL3_RT_HEADER, p, DTLS1_RT_HEADER_LENGTH, s, s->msg_callback_arg);

		/* Pull apart the header into the DTLS1_RECORD */
		rr->type= *(p++);
		ssl_major= *(p++);
		ssl_minor= *(p++);
		version=(ssl_major<<8)|ssl_minor;

		/* sequence number is 64 bits, with top 2 bytes = epoch */ 
		n2s(p,rr->epoch);

		memcpy(&(s->s3->read_sequence[2]), p, 6);
		p+=6;

		n2s(p,rr->length);

		/* Lets check version */
		if (!s->first_packet)
			{
			if (version != s->version)
				{
				/* unexpected version, silently discard */
				rr->length = 0;
				s->packet_length = 0;
				goto again;
				}
			}

		if ((version & 0xff00) != (s->version & 0xff00))
			{
			/* wrong version, silently discard record */
			rr->length = 0;
			s->packet_length = 0;
			goto again;
			}

		if (rr->length > SSL3_RT_MAX_ENCRYPTED_LENGTH)
			{
			/* record too long, silently discard it */
			rr->length = 0;
			s->packet_length = 0;
			goto again;
			}

		/* now s->rstate == SSL_ST_READ_BODY */
		}

	/* s->rstate == SSL_ST_READ_BODY, get and decode the data */

	if (rr->length > s->packet_length-DTLS1_RT_HEADER_LENGTH)
		{
		/* now s->packet_length == DTLS1_RT_HEADER_LENGTH */
		i=rr->length;
		n=ssl3_read_n(s,i,i,1);
		/* this packet contained a partial record, dump it */
		if ( n != i)
			{
			rr->length = 0;
			s->packet_length = 0;
			goto again;
			}

		/* now n == rr->length,
		 * and s->packet_length == DTLS1_RT_HEADER_LENGTH + rr->length */
		}
	s->rstate=SSL_ST_READ_HEADER; /* set state for later operations */

	/* match epochs.  NULL means the packet is dropped on the floor */
	bitmap = dtls1_get_bitmap(s, rr, &is_next_epoch);
	if ( bitmap == NULL)
		{
		rr->length = 0;
		s->packet_length = 0;  /* dump this record */
		goto again;   /* get another record */
		}

#ifndef OPENSSL_NO_SCTP
	/* Only do replay check if no SCTP bio */
	if (!BIO_dgram_is_sctp(SSL_get_rbio(s)))
  		{
#endif
		/* Check whether this is a repeat, or aged record.
		 * Don't check if we're listening and this message is
		 * a ClientHello. They can look as if they're replayed,
		 * since they arrive from different connections and
 		 * would be dropped unnecessarily.
 		 */
 		if (!(s->d1->listen && rr->type == SSL3_RT_HANDSHAKE &&
		    *p == SSL3_MT_CLIENT_HELLO) &&
 		    !dtls1_record_replay_check(s, bitmap))
 			{
 			rr->length = 0;
			s->packet_length=0; /* dump this record */
			goto again;     /* get another record */
			}
#ifndef OPENSSL_NO_SCTP
  		}
#endif

	/* just read a 0 length packet */
	if (rr->length == 0) goto again;

	/* If this record is from the next epoch (either HM or ALERT),
	 * and a handshake is currently in progress, buffer it since it
	 * cannot be processed at this time. However, do not buffer
	 * anything while listening.
	 */
	if (is_next_epoch)
		{
		if ((SSL_in_init(s) || s->in_handshake) && !s->d1->listen)
			{
			dtls1_buffer_record(s, &(s->d1->unprocessed_rcds), rr->seq_num);
			}
		rr->length = 0;
		s->packet_length = 0;
		goto again;
		}

	if (!dtls1_process_record(s))
		{
		rr->length = 0;
		s->packet_length = 0;  /* dump this record */
		goto again;   /* get another record */
		}

	return(1);

	}
