

python llava/eval/run_llava.py --model-name checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query  "<image>\nPlease read the text in this image and return the information in the following JSON format (note xxx is placeholder, if the information is not available in the image, put \"N/A\" instead).\n{\"class\": xxx, \"DLN\": xxx, \"DOB\": xxx, \"Name\": xxx, \"Address\": xxx, \"EXP\": xxx, \"ISS\": xxx, \"SEX\": xxx, \"HGT\": xxx, \"WGT\": xxx, \"EYES\": xxx, \"HAIR\": xxx, \"DONOR\": xxx}" \
    --image-file "images/license/pad.png" 


python llava/eval/run_llava.py --model-name checkpoints/llama2-13b-mmc4sub+coyo-finetune-cleandualflan+llava+sgpt-linearclip-e1-nose-run2 --conv-mode vicuna_v1_1 \
    --query "<image>\nPlease read the text in this image and return the information in the following JSON format (note xxx is placeholder, if the information is not available in the image, put "N/A" instead).\n{"Surname": xxx, "Given Name": xxx, "USCIS \#": xxx, "Category": xxx, "Country of Birth": xxx, "Date of Birth": xxx, "SEX": xxx, "Card Expires": xxx, "Resident Since": xxx}" \
    --image-file "images/greencard/raw.png" 


