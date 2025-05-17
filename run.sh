 model=${1:-gpt-4o-mini}
num_turns=${2:-5}
language=${3:-korean}
num_process=${4:-50}
batch_size=${5:-4}
cut_turns=${6:-$num_turns}

case "$model" in
  */*) model_clean=$(basename "$model") ;;
  *)   model_clean="$model" ;;
esac

echo "🤖 Model to evaluate: $model_clean"
echo "🚀 EPAG initiating..."

echo "🧪 Simulating dialogues..."
python3 simulate.py --model "$model" --num_turns "$num_turns" --language "$language" --num_process "$num_process" --batch_size "$batch_size"

echo "🧹 Organizing dialogues..."
python3 organize_cut.py --dialogue_file "${model_clean}_turn_${num_turns}.csv" --num_process "$num_process" --language "$language" --cut_turn "$cut_turns"

echo "🔍 Comparing results..."
python3 compare.py --dialogue_file "${model_clean}_turn_${cut_turns}.csv" --num_process "$num_process" --language "$language"

echo "🩺 Diagnosing..."
python3 diagnose.py --dialogue_file "${model_clean}_turn_${cut_turns}.csv"

echo "📊 Scoring..."
python3 score.py --dialogue_file "${model_clean}_turn_${cut_turns}.csv"

echo "✅ EPAG completed successfully!"
