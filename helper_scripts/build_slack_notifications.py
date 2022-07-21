import requests
import sys
import getopt

# send Slack Message Using Slack API


def send_slack_message(message):
    payload = '{"text":"%s"}' % message
    url = 'https://hooks.slack.com/services/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'
    response = requests.post(url, data=payload)
    print(response.text)


def main(argv):

    message = ' '

    try:
        opts, args = getopt.getopt(argv, "hm:", ["message="])
    except getopt.GetoptError:
        print('build_slack_notification.py -m <message>')
        sys.exit(2)
    if len(opts) == 0:
        message = "Hello, World"
    for opt, arg in opts:
        if opt == '-h':
            print('build_slack_notifications.py -m <message>')
            sys.exit()
        elif opt in ("-m", "--message"):
            message = arg

    send_slack_message(message)


if __name__ == '__main__':
    main(sys.argv[1:])



# def post_message_to_slack(text, blocks = None):
#     slack_token = 'xoxb-my-bot-token'
#     slack_channel = '#my-channel'
#     slack_icon_emoji = ':see_no_evil:'
#     slack_user_name = 'Double Images Monitor'
#
#     return requests.post('https://slack.com/api/chat.postMessage', {
#         'token': slack_token,
#         'channel': slack_channel,
#         'text': text,
#         'icon_emoji': slack_icon_emoji,
#         'username': slack_user_name,
#         'blocks': json.dumps(blocks) if blocks else None
#     }).json()


# def slack_message(message, channel):
#     token = '[YOUR TOKEN]'
#     sc = SlackClient(token)
#     sc.api_call('chat.postMessage', channel=channel,
#                 text=message, username='My Sweet Bot',
#                 icon_emoji=':robot_face:')

# if __name__ == '__main__':
#     # slack_message('Hello World Max', 'inversion-algorithm-development')
#
#     slack_info = 'There are *{}* double images detected for *{}* products. ' \
#                  'Please check the <https://{}.s3-eu-west-1.amazonaws.com/{}|Double Images Monitor>.'
#
#     post_message_to_slack(slack_info)
