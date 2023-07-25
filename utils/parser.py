import argparse
# 运行的代码参数

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=True)

parser.add_argument('--Device',
                    metavar='-D',
                    type=str,
                    required=False,
                    default='cpu',
                    help="cuda or cpu")

parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='STFT',
                    help="model name")

parser.add_argument('--freq', 
					metavar='-f', 
					type=str, 
					required=False,
					default='False',
                    help="convert to frequence")

parser.add_argument('--twenty',
                    action='store_true',
                    help="train using less data")

parser.add_argument('--test', 
					action='store_true', 
					help="test the model")

args = parser.parse_args()
