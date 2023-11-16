import json
import collections
import argparse
import random
import numpy as np
import requests
import re

# api key for query. see https://docs.together.ai/docs/get-started
def your_api_key():
    YOUR_API_KEY = '9e3c2abd5e605c5f5d36f6760aefb82246de307313704f78884784dc149537d0'
    return YOUR_API_KEY


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    #prefix = '''Question: what is 1234567+1234567?\nAnswer: 2469134\nQuestion: what is '''
    prefix = """Question: What is 1234567 + 7654321?\nAnswer: 8888888\nQuestion: What is 2345678 + 8765432?\nAnswer: 11111110\n
    Question: What is 5907670 + 8961742?\nAnswer: 14869412\nQuestion: What is 3528570 + 3195626?\nAnswer: 6724196\n
    Question: What is 1775918 + 9055957?\nAnswer: 10831875\nQuestion: What is 4783276 + 7484357?\nAnswer: 12267633\n
    Question: What is 2402853 + 6760529?\nAnswer: 9163382\nQuestion: What is 3791408 + 3033665?\nAnswer: 6825073\n
    Question: What is 2334899 + 3190456?\nAnswer: 5525355\nQuestion: What is 1111111 + 2222222?\nAnswer: 3333333\n
    Question: What is 9402239 + 9542415?\nAnswer: 18944654\nQuestion: What is 3224126 + 1488374?\nAnswer: 4712500\n
    Question: What is """

    suffix = '?\nAnswer: '

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys. 
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 50, # max_tokens must be >= 50 because we don't always have prior on output length 
        'temperature': 0.5,
        'top_k': 40,
        'top_p': 0.6,
        'repetition_penalty': 1,
        'stop': []}
    
    return config


def your_pre_processing(s):
    return s

    
def your_post_processing(output_string):
    """Returns the post processing function to extract the answer for addition
    Returns:
        For: the function returns extracted result
    Note:
        do not attempt to "hack" the post processing function
        by extracting the two given numbers and adding them.
        the autograder will check whether the post processing function contains arithmetic additiona and the graders might also manually check.
    """
    first_line = output_string.splitlines()[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res