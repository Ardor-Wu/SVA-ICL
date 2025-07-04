 static int non_vulnerable_func(struct x86_emulate_ctxt *ctxt)
 {
 	struct x86_emulate_ops *ops = ctxt->ops;
	struct desc_struct cs, ss;
	u64 msr_data;
	u16 cs_sel, ss_sel;
	u64 efer = 0;

	/* syscall is not available in real mode */
	if (ctxt->mode == X86EMUL_MODE_REAL ||
 	    ctxt->mode == X86EMUL_MODE_VM86)
 		return emulate_ud(ctxt);
 
 	ops->get_msr(ctxt, MSR_EFER, &efer);
 	setup_syscalls_segments(ctxt, &cs, &ss);
 
 	ops->get_msr(ctxt, MSR_STAR, &msr_data);
 	msr_data >>= 32;
 	cs_sel = (u16)(msr_data & 0xfffc);
	ss_sel = (u16)(msr_data + 8);

	if (efer & EFER_LMA) {
		cs.d = 0;
		cs.l = 1;
	}
	ops->set_segment(ctxt, cs_sel, &cs, 0, VCPU_SREG_CS);
	ops->set_segment(ctxt, ss_sel, &ss, 0, VCPU_SREG_SS);

	ctxt->regs[VCPU_REGS_RCX] = ctxt->_eip;
	if (efer & EFER_LMA) {
#ifdef CONFIG_X86_64
		ctxt->regs[VCPU_REGS_R11] = ctxt->eflags & ~EFLG_RF;

		ops->get_msr(ctxt,
			     ctxt->mode == X86EMUL_MODE_PROT64 ?
			     MSR_LSTAR : MSR_CSTAR, &msr_data);
		ctxt->_eip = msr_data;

		ops->get_msr(ctxt, MSR_SYSCALL_MASK, &msr_data);
		ctxt->eflags &= ~(msr_data | EFLG_RF);
#endif
	} else {
		/* legacy mode */
		ops->get_msr(ctxt, MSR_STAR, &msr_data);
		ctxt->_eip = (u32)msr_data;

		ctxt->eflags &= ~(EFLG_VM | EFLG_IF | EFLG_RF);
	}

	return X86EMUL_CONTINUE;
}
