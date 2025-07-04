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

static unsigned int fanout_demux_rollover(struct packet_fanout *f,
					  struct sk_buff *skb,
					  unsigned int idx, bool try_self,
					  unsigned int num)
{
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

	struct packet_sock *po, *po_next, *po_skip = NULL;
	unsigned int i, j, room = ROOM_NONE;

	po = pkt_sk(f->arr[idx]);

	if (try_self) {
		room = packet_rcv_has_room(po, skb);
		if (room == ROOM_NORMAL ||
		    (room == ROOM_LOW && !fanout_flow_is_huge(po, skb)))
			return idx;
		po_skip = po;
	}

	i = j = min_t(int, po->rollover->sock, num - 1);
	do {
		po_next = pkt_sk(f->arr[i]);
		if (po_next != po_skip && !po_next->pressure &&
		    packet_rcv_has_room(po_next, skb) == ROOM_NORMAL) {
			if (i != j)
				po->rollover->sock = i;
			atomic_long_inc(&po->rollover->num);
			if (room == ROOM_LOW)
				atomic_long_inc(&po->rollover->num_huge);
			return i;
		}

		if (++i == num)
			i = 0;
	} while (i != j);

	atomic_long_inc(&po->rollover->num_failed);
	return idx;
}
