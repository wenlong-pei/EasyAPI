import translators as ts

wyw_text = """
I have a Custom User model that takes user ip address. I want to add the IP address of the user upon completion of the sign up form. Where do I implement the below code? I am not sure whether to put this into my forms.py or views.py file.
I expect to be able to save the user's ip address into my custom user table upon sign up.
"""

print(ts.alibaba(wyw_text, from_language="en", to_language="zh")) # default: from_language='auto', to_language='en'