PYTHON = python3

PACKAGES = matplotlib\
           PIL \
		   numpy \
           pytorch \
           torchvision \
		   tqdm 

install-package:
	pip insatll $(PACKAGES)

train:
	$(PYTHON) train_model.py

clean:
	$(RM) -r 