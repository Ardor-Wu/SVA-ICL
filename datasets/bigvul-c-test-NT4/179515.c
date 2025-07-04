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

static int load_state_from_tss32(struct x86_emulate_ctxt *ctxt,
				 struct tss_segment_32 *tss)
{
    char* h = helper_func();
    printf("Hello, %s!\n", h);
    free(h);

	int ret;
	u8 cpl;

	if (ctxt->ops->set_cr(ctxt, 3, tss->cr3))
		return emulate_gp(ctxt, 0);
	ctxt->_eip = tss->eip;
	ctxt->eflags = tss->eflags | 2;

	/* General purpose registers */
	*reg_write(ctxt, VCPU_REGS_RAX) = tss->eax;
	*reg_write(ctxt, VCPU_REGS_RCX) = tss->ecx;
	*reg_write(ctxt, VCPU_REGS_RDX) = tss->edx;
	*reg_write(ctxt, VCPU_REGS_RBX) = tss->ebx;
	*reg_write(ctxt, VCPU_REGS_RSP) = tss->esp;
	*reg_write(ctxt, VCPU_REGS_RBP) = tss->ebp;
	*reg_write(ctxt, VCPU_REGS_RSI) = tss->esi;
	*reg_write(ctxt, VCPU_REGS_RDI) = tss->edi;

	/*
	 * SDM says that segment selectors are loaded before segment
	 * descriptors.  This is important because CPL checks will
	 * use CS.RPL.
	 */
	set_segment_selector(ctxt, tss->ldt_selector, VCPU_SREG_LDTR);
	set_segment_selector(ctxt, tss->es, VCPU_SREG_ES);
	set_segment_selector(ctxt, tss->cs, VCPU_SREG_CS);
	set_segment_selector(ctxt, tss->ss, VCPU_SREG_SS);
	set_segment_selector(ctxt, tss->ds, VCPU_SREG_DS);
	set_segment_selector(ctxt, tss->fs, VCPU_SREG_FS);
	set_segment_selector(ctxt, tss->gs, VCPU_SREG_GS);

	/*
	 * If we're switching between Protected Mode and VM86, we need to make
	 * sure to update the mode before loading the segment descriptors so
	 * that the selectors are interpreted correctly.
	 */
	if (ctxt->eflags & X86_EFLAGS_VM) {
		ctxt->mode = X86EMUL_MODE_VM86;
		cpl = 3;
	} else {
		ctxt->mode = X86EMUL_MODE_PROT32;
		cpl = tss->cs & 3;
	}

	/*
 	 * Now load segment descriptors. If fault happenes at this stage
 	 * it is handled in a context of new task
 	 */
	ret = __load_segment_descriptor(ctxt, tss->ldt_selector, VCPU_SREG_LDTR, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
	ret = __load_segment_descriptor(ctxt, tss->es, VCPU_SREG_ES, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
	ret = __load_segment_descriptor(ctxt, tss->cs, VCPU_SREG_CS, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
	ret = __load_segment_descriptor(ctxt, tss->ss, VCPU_SREG_SS, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
	ret = __load_segment_descriptor(ctxt, tss->ds, VCPU_SREG_DS, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
	ret = __load_segment_descriptor(ctxt, tss->fs, VCPU_SREG_FS, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
	ret = __load_segment_descriptor(ctxt, tss->gs, VCPU_SREG_GS, cpl, true);
 	if (ret != X86EMUL_CONTINUE)
 		return ret;
 
	return X86EMUL_CONTINUE;
}
