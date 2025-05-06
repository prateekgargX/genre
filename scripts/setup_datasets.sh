mkdir -p datasets
cd datasets/
git clone https://github.com/carla-recourse/cf-data.git
mkdir gmsc
cp cf-data/give_me_some_credit_train.csv ./gmsc/train.csv
cp cf-data/give_me_some_credit_test.csv ./gmsc/test.csv

# mkdir adult
# cp cf-data/adult_train.csv ./adult/train.csv
# cp cf-data/adult_test.csv ./adult/test.csv

# mkdir compas
# cp cf-data/compas_train.csv ./compas/train.csv
# cp cf-data/compas_test.csv ./compas/test.csv

# mkdir heloc
# cp cf-data/heloc_train.csv ./heloc/train.csv
# cp cf-data/heloc_test.csv ./heloc/test.csv

rm -rf cf-data
