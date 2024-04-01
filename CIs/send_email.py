import os
import os.path as osp
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart


sender = os.environ.get("VILA_CI_SENDER", None)
password = os.environ.get("VILA_CI_PASSWORD", None)
# recipients = ["ligengz@nvidia.com", "jasonlu@nvidia.com"]

assert sender is not None and password is not None, "sender account and password must be set."
# recipients = os.environ.get("VILA_CI_RECIPIENTS", "ligengz@nvidia.com")
# recipients = recipients.split(",")


def send_email(subject, body, sender, recipients, password, files=None):

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    
    msg.attach(MIMEText(body, "html"))    
    # send error files
    for f in files or []:
        fpath = osp.join("dev", f.replace("/", "--") + ".err")
        print(f"uploading {fpath}")
        with open(fpath, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=osp.basename(fpath.replace(".py.err", ".txt"))
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % osp.basename(fpath.replace(".py.err", ".txt"))
        msg.attach(part)
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())
    print(f"Message sent to {recipients}")


def main(
    text=None,
    recipients="ligengz@nvidia.com",
):
    today = datetime.now().strftime("%m/%d/%Y %H:%M")
    subject = f"[VILA] Continual Test Report {today}"
    body = f"""Testing"""
    
    failed_jobs = []
    if text is not None:
        body_md = open(text, "r").readlines()

        info_html = ""
        
        info_success = []
        info_failed = []
        for md in body_md:
            if "failed" in md:
                info = f"""<li><span style="color: red;">{md}</span></li>\n"""
                failed_jobs.append(md.replace("[failed]", "").strip())
                info_failed.append(info)
            else:
                info = f"""<li><span style="color: green;">{md}</span></li>\n"""
                info_success.append(info)
        info = "".join(info_failed + info_success)
        
        if len(info_failed) > 0:
            header = "<p>The log for failed jobs are attached. You can run <b> python xxx.py </b> to reproduce.</p>"
        else:
            header = "All checks have passed succesfully!"
        
        body = f"""\
            <html>
            <body>
                {header}
                <ul>
                    {info}
                </ul>
            </body>
            </html>
            """
    # recipients = os.environ.get("VILA_CI_RECIPIENTS", "ligengz@nvidia.com")
    recipients = recipients.split(",")

    send_email(subject, body, sender, recipients, password, files=failed_jobs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
