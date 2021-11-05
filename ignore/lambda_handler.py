import boto3

region = 'ap-northeast-2'
instances = ['i-0b243b03eaea146a0']
ec2 = boto3.client('ec2', region_name = region)

def lambda_handler(event, context):
    ec2.start_instances(Instance=instances)
    print('started your instance' + str(instances))