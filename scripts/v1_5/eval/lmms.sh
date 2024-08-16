TASK=$1
MODEL_PATH=$2
CONV_MODE=$3

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/lmms-$TASK"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}

export LMMS_EVAL_PLUGINS=llava.eval.lmms
torchrun --nproc_per_node=$NPROC_PER_NODE \
	-m lmms_eval \
	--model vila_internal \
	--model_args model_path=$MODEL_PATH,conv_mode=$CONV_MODE \
	--tasks $TASK \
	--log_samples \
	--output_path $OUTPUT_DIR

mv $OUTPUT_DIR/*_vila_internal_*/* $OUTPUT_DIR
rm -r $OUTPUT_DIR/*_vila_internal_*
rm -r $OUTPUT_DIR/rank*_metric_eval_done.txt
