import os, sys, os.path as osp
import smtplib
from email.mime.text import MIMEText

sender = os.environ.get("VILA_CI_SENDER", None)
password = os.environ.get("VILA_CI_PASSWORD", None)
# recipients = ["ligengz@nvidia.com", "jasonlu@nvidia.com"]
recipients = os.environ.get("VILA_CI_RECIPIENTS", "")

assert sender is not None and password is not None, "sender account and password must be set."

recipients = recipients.split(",")

def send_email(subject, body, sender, recipients, password):
    msg = MIMEText(body, "html")
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
       smtp_server.login(sender, password)
       smtp_server.sendmail(sender, recipients, msg.as_string())
    print("Message sent!")

def main(text=None):
    from datetime import datetime
    today = datetime.now().strftime("%m/%d/%Y %H:%M")
    subject = f"[Sample-Dev] VILA Regression Report {today}"
    body = f"""Testing"""
    if text is not None:
        body_md = open(text, "r").readlines()
        
        info = ""
        for md in body_md:
            if "failed" in md:
                info += f'''<li><span style="color: red;">{md}.</span></li>\n'''
            else:
                info += f'''<li><span style="color: green;">{md}.</span></li>\n'''
                
        body = f"""\
            <html>
            <body>
                <ul>
                    {info}
                </ul>
            </body>
            </html>
            """
                
        
    send_email(subject, body, sender, recipients, password)

if __name__ == "__main__":
    import fire
    fire.Fire(main)