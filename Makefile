.PHONY: all paper previews view clean

all: paper

paper:
	$(MAKE) -C graph_ml_final all

previews:
	$(MAKE) -C graph_ml_final previews

view:
	$(MAKE) -C graph_ml_final view

clean:
	$(MAKE) -C graph_ml_final clean
