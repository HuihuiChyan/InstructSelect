for FILE_NAME in "sharegpt-lora8max100k-maxrwd10k";
do
    LLMJUDGE_DIR="./mt_bench_eval/llm_judge"
    JUDGE_FILE="./mt_bench_eval/llm_judge/judge_prompts.jsonl"
    ALPACA_FILE="./sharegpt_data/$FILE_NAME-alpaca-farm.json"
    ALPACA_DIR=""
    # python -u mt_bench_eval/gen_judgement.py \
    #      --model-list $FILE_NAME-mt-bench \
    #      --data-dir $LLMJUDGE_DIR \
    #      --judge-file $JUDGE_FILE \
    #      --parallel 2

    # python mt_bench_eval/show_result.py \
    #     --input-file $LLMJUDGE_DIR"/mt_bench/model_judgment/gpt-4_single.jsonl"

    LLMJUDGE_DIR="./mt_bench_eval/llm_judge"
    JUDGE_FILE="./mt_bench_eval/llm_judge/judge_prompts.jsonl"
    ALPACA_FILE="./sharegpt_data/$FILE_NAME-alpaca-farm.json"
    ALPACA_DIR=""
    # python -u mt_bench_eval/gen_judgement.py \
    #      --model-list $FILE_NAME-vicuna-bench \
    #      --data-dir $LLMJUDGE_DIR \
    #      --judge-file $JUDGE_FILE \
    #      --parallel 2 \
    #      --bench-name vicuna_bench
    
    python mt_bench_eval/show_result.py \
        --input-file $LLMJUDGE_DIR"/vicuna_bench/model_judgment/gpt-4_single.jsonl"
  

    # python -m alpaca_eval.main evaluate \
    #   --model_outputs $ALPACA_FILE \
    #   --output_path ./alpaca_eval/ \
    #   --precomputed_leaderboard ./alpaca_eval/leaderboard.csv
done