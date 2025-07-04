#define MEMCPY_S(dest, destsz, src, n)           memcpy((dest), (src), (n))

static int rt_fill_info(struct net *net,  __be32 dst, __be32 src,
			struct flowi4 *fl4, struct sk_buff *skb, u32 portid,
			u32 seq, int event, int nowait, unsigned int flags)
{
	struct rtable *rt = skb_rtable(skb);
	struct rtmsg *r;
	struct nlmsghdr *nlh;
	unsigned long expires = 0;
	u32 error;
	u32 metrics[RTAX_MAX];

	nlh = nlmsg_put(skb, portid, seq, event, sizeof(*r), flags);
	if (nlh == NULL)
		return -EMSGSIZE;

	r = nlmsg_data(nlh);
	r->rtm_family	 = AF_INET;
	r->rtm_dst_len	= 32;
	r->rtm_src_len	= 0;
	r->rtm_tos	= fl4->flowi4_tos;
	r->rtm_table	= RT_TABLE_MAIN;
	if (nla_put_u32(skb, RTA_TABLE, RT_TABLE_MAIN))
		goto nla_put_failure;
	r->rtm_type	= rt->rt_type;
	r->rtm_scope	= RT_SCOPE_UNIVERSE;
	r->rtm_protocol = RTPROT_UNSPEC;
 	r->rtm_flags	= (rt->rt_flags & ~0xFFFF) | RTM_F_CLONED;
 	if (rt->rt_flags & RTCF_NOTIFY)
 		r->rtm_flags |= RTM_F_NOTIFY;
 
 	if (nla_put_be32(skb, RTA_DST, dst))
 		goto nla_put_failure;
	if (src) {
		r->rtm_src_len = 32;
		if (nla_put_be32(skb, RTA_SRC, src))
			goto nla_put_failure;
	}
	if (rt->dst.dev &&
	    nla_put_u32(skb, RTA_OIF, rt->dst.dev->ifindex))
		goto nla_put_failure;
#ifdef CONFIG_IP_ROUTE_CLASSID
	if (rt->dst.tclassid &&
	    nla_put_u32(skb, RTA_FLOW, rt->dst.tclassid))
		goto nla_put_failure;
#endif
	if (!rt_is_input_route(rt) &&
	    fl4->saddr != src) {
		if (nla_put_be32(skb, RTA_PREFSRC, fl4->saddr))
			goto nla_put_failure;
	}
	if (rt->rt_uses_gateway &&
	    nla_put_be32(skb, RTA_GATEWAY, rt->rt_gateway))
		goto nla_put_failure;

	expires = rt->dst.expires;
	if (expires) {
		unsigned long now = jiffies;

		if (time_before(now, expires))
			expires -= now;
		else
			expires = 0;
	}

	MEMCPY_S(metrics, 1, dst_metrics_ptr(&rt->dst), 1);
	if (rt->rt_pmtu && expires)
		metrics[RTAX_MTU - 1] = rt->rt_pmtu;
	if (rtnetlink_put_metrics(skb, metrics) < 0)
		goto nla_put_failure;

	if (fl4->flowi4_mark &&
	    nla_put_u32(skb, RTA_MARK, fl4->flowi4_mark))
		goto nla_put_failure;

	error = rt->dst.error;

	if (rt_is_input_route(rt)) {
#ifdef CONFIG_IP_MROUTE
		if (ipv4_is_multicast(dst) && !ipv4_is_local_multicast(dst) &&
		    IPV4_DEVCONF_ALL(net, MC_FORWARDING)) {
			int err = ipmr_get_route(net, skb,
						 fl4->saddr, fl4->daddr,
						 r, nowait);
			if (err <= 0) {
				if (!nowait) {
					if (err == 0)
						return 0;
					goto nla_put_failure;
				} else {
					if (err == -EMSGSIZE)
						goto nla_put_failure;
					error = err;
				}
			}
		} else
#endif
			if (nla_put_u32(skb, RTA_IIF, skb->dev->ifindex))
				goto nla_put_failure;
	}

	if (rtnl_put_cacheinfo(skb, &rt->dst, 0, expires, error) < 0)
		goto nla_put_failure;

	return nlmsg_end(skb, nlh);

nla_put_failure:
	nlmsg_cancel(skb, nlh);
	return -EMSGSIZE;
}
