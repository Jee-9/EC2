import boto3

region = 'ap-northeast-2'
instances = ['0b243b03eaea146a0']
ec2 = boto3.client('ec2', region_name=region)

def lambda_handler(event, context):
    ec2.stop_instances(Instance=instances)
    print('stopped your instance' + str(instances))