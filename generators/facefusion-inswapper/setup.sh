#!/usr/bin/env bash
# Reuses the facefusion install (its setup.sh downloads this method's models too).
. "$(dirname "$0")/../common.sh"
reuse facefusion
done_msg
