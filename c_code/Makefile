all: input.aplx sum.aplx threshold.aplx weight.aplx

%.aplx: %.mk %.c
	"$(MAKE)" -f $<

tidy:
	for d in input sum threshold weight; \
		do ("$(MAKE)" -f $$d.mk tidy) || exit $$?; done

clean:
	for d in input sum threshold weight; \
		do ("$(MAKE)" -f $$d.mk clean) || exit $$?; done
