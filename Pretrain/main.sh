# Set `model' to `orig', `RC', `LBE', `IP' and `MIB' for different models.
# Set `dataset' for different pre-training datasets.
# Set `testset' for different transfer datasets.


# Training
## SimCLR
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --dataset CIFAR10
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --dataset STL-10

# Linear Evaluation
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --projection_dim 128 --dataset CIFAR10 --testset cu_birds
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --projection_dim 128 --dataset STL-10 --testset cu_birds
