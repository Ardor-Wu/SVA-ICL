 static int set_register(pegasus_t *buffer1, __u16 indx, __u8 data)
 {
 	int ret;
 
 	ret = usb_control_msg(buffer1->usb, usb_sndctrlpipe(buffer1->usb, 0),
 			      PEGASUS_REQ_SET_REG, PEGASUS_REQT_WRITE, data,
			      indx, &data, 1, 1000);
 	if (ret < 0)
 		netif_dbg(buffer1, drv, buffer1->net,
 			  "%s returned %d\n", __func__, ret);
 	return ret;
 }
