all: input.aplx sum.aplx threshold.aplx weight.aplx

%.aplx: %.Makefile %.c
	"$(MAKE)" -f $<

clean:
	for d in input sum threshold weight; \
		do ("$(MAKE)" -f $$d.Makefile clean) || exit $$?; done
