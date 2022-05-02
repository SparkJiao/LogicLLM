#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p JX-GPU-IB
#SBATCH --mem 40G
#SBATCH --gpus-per-task 1

input_file=AR-LSAT/data/AR_TrainingData.json

## Predict on passage.
#for iter_id in 0 1 2 3 4; do
#  srun python -u predictor_gen.py dev_file=${input_file} read_tensor.iter_id=${iter_id} -cp conf/t5/ar-lsat/prediction -cn t5_large_proofwriter_iter_v1_0
#  srun python -u preprocess/join_gen_predictions_iter.py --input_file ${input_file} \
#   --predictions AR-LSAT/data/t5.large.proof_writer.ft.v1.0/train-step22000-False-${iter_id}/prediction.gen.best.dev/generate-predictions.pt --iter_id ${iter_id}
#  input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/train-step22000-False-${iter_id}/prediction.gen.best.dev/combine_${iter_id}.json
#done;

input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/train-step22000-False-4/prediction.gen.best.dev/combine_4.json

## Predict on passage.
#for iter_id in 0 1 2 3 4; do
#  srun python -u predictor_gen.py dev_file=${input_file} read_tensor.iter_id=${iter_id} read_tensor.predict_on_question=True -cp conf/t5/ar-lsat/prediction -cn t5_large_proofwriter_iter_v1_0
#  srun python -u preprocess/join_gen_predictions_iter.py --input_file ${input_file} \
#   --predictions AR-LSAT/data/t5.large.proof_writer.ft.v1.0/train-step22000-True-${iter_id}/prediction.gen.best.dev/generate-predictions.pt --iter_id ${iter_id}
#  input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/train-step22000-True-${iter_id}/prediction.gen.best.dev/combine_${iter_id}.json
#done;


input_file=AR-LSAT/data/AR_DevelopmentData.json

# Predict on passage.
for iter_id in 0 1 2 3 4; do
  prediction_dir=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-False-${iter_id}/
  srun python -u predictor_gen.py dev_file=${input_file} read_tensor.iter_id=${iter_id} prediction_dir=${prediction_dir} -cp conf/t5/ar-lsat/prediction -cn t5_large_proofwriter_iter_v1_0
  srun python -u preprocess/join_gen_predictions_iter.py --input_file ${input_file} \
   --predictions AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-False-${iter_id}/prediction.gen.best.dev/generate-predictions.pt --iter_id ${iter_id}
  input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-False-${iter_id}/prediction.gen.best.dev/combine_${iter_id}.json
done;


input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-False-4/prediction.gen.best.dev/combine_4.json

# Predict on passage.
for iter_id in 0 1 2 3 4; do
  prediction_dir=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-True-${iter_id}/
  srun python -u predictor_gen.py dev_file=${input_file} read_tensor.iter_id=${iter_id} prediction_dir=${prediction_dir} read_tensor.predict_on_question=True -cp conf/t5/ar-lsat/prediction -cn t5_large_proofwriter_iter_v1_0
  srun python -u preprocess/join_gen_predictions_iter.py --input_file ${input_file} \
   --predictions AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-True-${iter_id}/prediction.gen.best.dev/generate-predictions.pt --iter_id ${iter_id}
  input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/dev-step22000-True-${iter_id}/prediction.gen.best.dev/combine_${iter_id}.json
done;



input_file=AR-LSAT/data/AR_TestData.json

# Predict on passage.
for iter_id in 0 1 2 3 4; do
  prediction_dir=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-False-${iter_id}/
  srun python -u predictor_gen.py dev_file=${input_file} read_tensor.iter_id=${iter_id} prediction_dir=${prediction_dir} -cp conf/t5/ar-lsat/prediction -cn t5_large_proofwriter_iter_v1_0
  srun python -u preprocess/join_gen_predictions_iter.py --input_file ${input_file} \
   --predictions AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-False-${iter_id}/prediction.gen.best.dev/generate-predictions.pt --iter_id ${iter_id}
  input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-False-${iter_id}/prediction.gen.best.dev/combine_${iter_id}.json
done;


input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-False-4/prediction.gen.best.dev/combine_4.json

# Predict on passage.
for iter_id in 0 1 2 3 4; do
  prediction_dir=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-True-${iter_id}/
  srun python -u predictor_gen.py dev_file=${input_file} read_tensor.iter_id=${iter_id} prediction_dir=${prediction_dir} read_tensor.predict_on_question=True -cp conf/t5/ar-lsat/prediction -cn t5_large_proofwriter_iter_v1_0
  srun python -u preprocess/join_gen_predictions_iter.py --input_file ${input_file} \
   --predictions AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-True-${iter_id}/prediction.gen.best.dev/generate-predictions.pt --iter_id ${iter_id}
  input_file=AR-LSAT/data/t5.large.proof_writer.ft.v1.0/test-step22000-True-${iter_id}/prediction.gen.best.dev/combine_${iter_id}.json
done;

