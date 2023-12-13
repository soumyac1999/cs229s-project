#!/bin/bash

rm -f cs229s_submission.zip
zip -r cs229s_submission.zip . -x "data/*" -x "examples/*" -x "out/*" -x "shakespeare/*" -x "wandb/*" -x "wikitext/*" -x ".git/*"
