#!/bin/sh
set -eu
printf '%s\n' 'BLOCKED: 原训练入口用正式测试集选检查点；先经理论审查确定训练/验证拆分并实现独立验证后再运行。' >&2
exit 1
