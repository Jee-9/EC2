
## python
# instance 중단 코드

import boto3
region = 'ap-northeast-2'
instances = []
ec2 = boto3.client('ec2', region_name = region)

def lambda_handler(event, context):
    ec2.stop_instances(Instances=instances)
    print('stopped your instance' + str(instances))


# instance turn-on 코드
import boto3
region = 'ap-notrheast-2'
instances = []
ec2 = boto3.client('ec2', region_name = region)

def lambda_handler(event, context):
    ec2.start_instances(Instances = instances)
    print('started your instance' + str(instances))

    
