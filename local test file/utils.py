import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()


def get_connection():
    """Establish MySQL connection using environment variables."""
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "localhost"),
        port=int(os.getenv("MYSQLPORT", 3306)),
        user=os.getenv("MYSQLUSER", "root"),
        password=os.getenv("MYSQLPASSWORD", ""),
        database=os.getenv("MYSQLDATABASE", "mscis_project1"),
    )


def create_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS testcase (
            s_no INT AUTO_INCREMENT PRIMARY KEY,
            sender VARCHAR(255),
            subject VARCHAR(255),
            body TEXT,
            label TINYINT
        )
    """
    )


def insert_testcases(cursor):

    test_data = [
        # ========== 35 Legitimate emails (label = 0) ==========
        (
            "no-reply@amazon.in",
            "Your Amazon.in order has shipped",
            "Hi, your order #AMZ12345 has left our warehouse and is on its way. Track it at your Orders page.",
            0,
        ),
        (
            "tickets@irctc.co.in",
            "E-Ticket Confirmed: Train 12627",
            "Dear Passenger, your ticket for train 12627 on 2025-10-05 is confirmed. PNR: 1234567890. Carry a valid ID.",
            0,
        ),
        (
            "support@paytm.com",
            "Payment Received: ₹2,499",
            "We have received your payment of ₹2,499 to Flipkart. Transaction ID: TXN987654. Thank you for using Paytm.",
            0,
        ),
        (
            "alerts@icicibank.com",
            "Debit Alert: ₹3,200",
            "Your account ending 6789 was debited by ₹3,200 at POS/Online on 2025-08-22. If unauthorized, contact us immediately.",
            0,
        ),
        (
            "newsletter@netflix.com",
            "Top picks for you this week",
            "Hi — we picked shows based on your watch history. Open the Netflix app to explore personalized recommendations.",
            0,
        ),
        (
            "noreply@airindia.in",
            "Flight Booking Confirmation - AI 512",
            "Your flight AI 512 on 2025-09-10 is confirmed. E-ticket attached. Please arrive 2 hours before departure.",
            0,
        ),
        (
            "hr@xyzcorp.com",
            "Interview Scheduled: Software Engineer",
            "Congratulations — we would like to invite you for a virtual interview on Monday at 11:00 AM. Zoom link included.",
            0,
        ),
        (
            "support@flipkart.com",
            "Return Approved for Order #FLK9876",
            "Your return request for order FLK9876 has been approved. Refund will be processed to your original payment method.",
            0,
        ),
        (
            "billing@spotify.com",
            "Receipt for your Spotify Subscription",
            "Thanks for renewing. Your invoice for the Premium plan is attached. Visit your account for details.",
            0,
        ),
        (
            "service@ola.in",
            "Ride Invoice: Ola Share",
            "Your ride with Ola on 2025-08-20 is complete. Total fare ₹120. View detailed invoice in the app.",
            0,
        ),
        (
            "registration@university.edu",
            "Welcome to Orientation Week",
            "Dear student, welcome! Orientation schedule and campus map are attached. See you on campus on 2025-09-01.",
            0,
        ),
        (
            "support@sbi.co.in",
            "NetBanking Login: New Device Added",
            "A new device was added to your SBI netbanking session from Mumbai. If this was you, no action needed.",
            0,
        ),
        (
            "alerts@gmail.com",
            "New Sign-in to your Google Account",
            "Someone signed in to your Google account from New Delhi. If this wasn't you, secure your account here: https://myaccount.google.com/security",
            0,
        ),
        (
            "noreply@uber.com",
            "Your trip receipt",
            "Thanks for riding with Uber. Your receipt for trip on 2025-08-19 is available in the app under 'Trips'.",
            0,
        ),
        (
            "support@zomato.com",
            "Order Delivered: Order #ZMT4455",
            "Your Zomato order has been delivered. Rate your experience and help us improve!",
            0,
        ),
        (
            "care@jobportal.com",
            "Application Received: Software Engineer",
            "Thank you for applying. Your application has been received and is under review. We'll update you within 2 weeks.",
            0,
        ),
        (
            "notification@gov.in",
            "Income Tax: Acknowledgement of e-Filing",
            "We have received your ITR acknowledgment for AY 2025-26. Log in to the e-filing portal for details.",
            0,
        ),
        (
            "support@dhl.com",
            "Shipment Update: Your Parcel is Out for Delivery",
            "Your parcel tracking #DHL1234567 is out for delivery today. Please ensure someone is available to receive it.",
            0,
        ),
        (
            "alerts@axisbank.com",
            "Credit Card Payment Successful",
            "Your credit card payment for ₹5,000 has been received. Thank you for paying on time.",
            0,
        ),
        (
            "noreply@linkedin.com",
            "Someone viewed your profile",
            "Good news — someone from ABC Corp viewed your profile. Visit LinkedIn to learn more.",
            0,
        ),
        (
            "support@apple.com",
            "Your AppleCare Plan Renewal",
            "Your AppleCare plan will renew on 2025-09-15. If you wish to cancel, visit your Apple ID subscriptions.",
            0,
        ),
        (
            "orders@bigbasket.com",
            "BigBasket: Order Confirmation #BB1234",
            "Thanks for shopping. Your BigBasket order #BB1234 will be delivered between 2-4 PM tomorrow.",
            0,
        ),
        (
            "no-reply@govpassport.in",
            "Passport Application - Appointment Confirmed",
            "Your appointment for passport application on 2025-09-03 is confirmed. Please bring required documents.",
            0,
        ),
        (
            "support@microsoft.com",
            "Office 365 Subscription Invoice",
            "Your invoice for Office 365 has been paid. Visit your Microsoft account for payment history.",
            0,
        ),
        (
            "noreply@booking.com",
            "Hotel Booking Confirmed - Booking #BK98765",
            "Your hotel at Mumbai is confirmed for 2025-10-12. Check-in from 14:00. Contact the hotel for special requests.",
            0,
        ),
        (
            "updates@twitter.com",
            "Your account summary",
            "Here's a summary of your account activity this week. Log in to view detailed insights.",
            0,
        ),
        (
            "care@govutility.in",
            "Water Bill Paid Successfully",
            "We have received your payment for the water bill (Account 12345). Thank you.",
            0,
        ),
        (
            "support@edx.org",
            "Course Certificate Available",
            "Congratulations! Your certificate for 'Machine Learning Basics' is now available to download.",
            0,
        ),
        (
            "noreply@schoolsystem.edu",
            "Parent-Teacher Meeting Scheduled",
            "The parent-teacher meeting for Grade 8 is scheduled on 2025-09-07 at 4 PM in the school auditorium.",
            0,
        ),
        (
            "alerts@hdfcbank.com",
            "OTP for Transaction",
            "Your OTP for transaction ID TXN9876 is 482910. This code will expire in 5 minutes.",
            0,
        ),
        (
            "support@gmail.com",
            "Security Tip: Review Connected Apps",
            "We recommend reviewing third-party apps connected to your Google account. Visit your account settings to manage access.",
            0,
        ),
        # ========== 35 Phishing emails (label = 1) ==========
        (
            "security@amaz0n-support.com",
            "Important: Verify Your Amazon Account Now",
            "We detected suspicious activity on your Amazon account. Verify within 24 hours to avoid suspension: http://verify-amazon.example/login",
            1,
        ),
        (
            "admin@paypal-update.com",
            "Your PayPal Account Has Been Limited",
            "We have limited access to your PayPal account. Click to restore access immediately: http://paypal-secure.example",
            1,
        ),
        (
            "support@icici-secure.com",
            "Urgent: Confirm Your KYC Documents",
            "Your account will be frozen unless you upload KYC documents here: http://icici-verify.example/upload",
            1,
        ),
        (
            "alerts@netfl1x-billing.com",
            "Payment Failed - Update Billing Info",
            "Your payment method was declined. Update billing now to continue streaming: http://netflix-bill.example",
            1,
        ),
        (
            "notify@apple-supports.com",
            "Apple ID Suspended",
            "We noticed suspicious activity. Reactivate your Apple ID by confirming your credentials: http://apple-id.example/confirm",
            1,
        ),
        (
            "service@google-account.com",
            "Unusual Sign-in Attempt",
            "We blocked a sign-in attempt. Confirm your identity here: http://google-secure.example/attest",
            1,
        ),
        (
            "secure@hdfc-update.com",
            "Verify Your Bank Details",
            "Your HDFC account shows inconsistencies. Verify immediately to prevent closure: http://hdfc-verify.example",
            1,
        ),
        (
            "support@flipkart-pay.com",
            "Refund Held - Action Required",
            "Your refund is on hold. Provide your card details to process refund: http://flipkart-refund.example",
            1,
        ),
        (
            "care@linkedin-team.com",
            "Confirm Your Identity",
            "We need to verify your LinkedIn account. Submit details here: http://linkedin-verify.example",
            1,
        ),
        (
            "noreply@ubs-bank.com",
            "Account Alert: Unauthorized Withdrawal",
            "An unauthorized withdrawal was detected. Log in to dispute the charge: http://ubs-fake.example/login",
            1,
        ),
        (
            "offers@paytm-cash.com",
            "You Won Cashback ₹5000",
            "Congratulations! Claim your cashback now by entering your card details: http://paytm-offer.example/claim",
            1,
        ),
        (
            "security@facebook-login.com",
            "Your Facebook Account Will Be Deleted",
            "We will delete your account unless you confirm ownership here: http://facebook-verify.example",
            1,
        ),
        (
            "help@airindia-bookings.com",
            "Flight Cancellation Notice",
            "Your booking AI345 has been cancelled due to payment issue. Reconfirm details here: http://airindia-fake.example/reconfirm",
            1,
        ),
        (
            "support@uber-payments.com",
            "Payment Issue - Immediate Action",
            "Your recent payment could not be processed. Provide card details here to avoid service interruption: http://uber-bill.example",
            1,
        ),
        (
            "alerts@icicibank-secure.com",
            "Validate Recent Transaction",
            "Confirm recent transactions now or your account will be blocked: http://icici-secure.example/validate",
            1,
        ),
        (
            "verify@gov-services.com",
            "Your PAN needs verification",
            "Your PAN details could not be validated. Update them immediately: http://gov-pan.example/verify",
            1,
        ),
        (
            "jobs@workfromhome-offer.com",
            "Work From Home - Earn ₹10,000 weekly",
            "Start earning immediately. Pay small registration fee and get access: http://wfh-scam.example/signup",
            1,
        ),
        (
            "noreply@amazon-prize.com",
            "You are a lucky winner!",
            "You have been selected to win a gift card. Click to claim: http://amazon-prize.example/claim",
            1,
        ),
        (
            "support@gmail-security.com",
            "Mailbox Quota Exceeded",
            "Your mailbox exceeded storage. Confirm to increase quota here: http://gmail-storage.example",
            1,
        ),
        (
            "alerts@paytmsecure.com",
            "OTP Verification Required",
            "To verify recent activity, enter your OTP here: http://paytm-verify.example",
            1,
        ),
        (
            "admin@office365-support.com",
            "Your Office 365 License Has Expired",
            "Renew your Office 365 license now to keep access: http://office365-renew.example",
            1,
        ),
        (
            "security@dhl-delivery.com",
            "Delivery Failed - Action Required",
            "We couldn't deliver your parcel. Pay customs here to release package: http://dhl-customs.example",
            1,
        ),
        (
            "help@bookmyshow-payments.com",
            "Ticket Payment Failed",
            "Your ticket payment failed. Update payment details here: http://bookmyshow-pay.example",
            1,
        ),
        (
            "alert@upi-update.com",
            "UPI Mobile Verified: Action Needed",
            "Your UPI ID verification failed. Confirm banking details here: http://upi-verify.example",
            1,
        ),
        (
            "service@instagram-support.com",
            "Verify Your Account to Avoid Suspension",
            "Your Instagram account will be suspended unless you verify via this form: http://insta-verify.example",
            1,
        ),
        (
            "claims@insurance-update.com",
            "Insurance Claim - Upload Documents",
            "To process your claim, upload scanned documents here: http://insurance-upload.example",
            1,
        ),
        (
            "security@bankofindia-alerts.com",
            "Immediate Verification Required",
            "We detected suspicious transfers. Verify now to avoid account hold: http://bankofindia-secure.example",
            1,
        ),
        (
            "offers@free-gifts.com",
            "Claim Your Free iPhone",
            "Limited time offer. Enter name and card details to claim: http://free-iphone.example/claim",
            1,
        ),
        (
            "support@zoom-meetings.com",
            "Meeting Recording Available - Login Required",
            "A recording is available but requires login here: http://zoom-records.example/login",
            1,
        ),
        (
            "notify@eduportal-update.com",
            "Exam Results Locked - Verify",
            "Your student portal is locked. Verify with credentials: http://edu-portal.example/unlock",
            1,
        ),
        (
            "alerts@reliance-pay.com",
            "Transaction Declined",
            "Your transaction was declined. Re-enter details to retry: http://reliance-pay.example/retry",
            1,
        ),
        (
            "admin@netflix-account.com",
            "Account Verification Needed",
            "We need to verify your payment method. Please update here: http://netflix-check.example",
            1,
        ),
    ]

    cursor.executemany(
        "INSERT INTO testcase (sender, subject, body, label) VALUES (%s, %s, %s, %s)",
        test_data,
    )


def main():
    connection = get_connection()
    cursor = connection.cursor()

    create_table(cursor)
    insert_testcases(cursor)

    connection.commit()
    cursor.close()
    connection.close()
    print("✅ Inserted 30 test cases into 'testcase' table.")


if __name__ == "__main__":
    main()


# # clean_dataset.py
# import pandas as pd
# import re

# DATA_PATH = r"D:\mscis\dataset\consolidated_emails.csv"
# SAVE_PATH = r"D:\mscis\dataset\cleaned_emails.csv"

# # Load dataset
# df = pd.read_csv(DATA_PATH)

# # Drop duplicates
# df = df.drop_duplicates()

# # Remove rows with missing label
# df = df.dropna(subset=["label"])

# # Fill missing subject/body with empty string
# df["subject"] = df["subject"].fillna("")
# df["body"] = df["body"].fillna("")

# # Combine subject + body into single column
# df["text"] = df["subject"] + " " + df["body"]


# # Optional: clean text
# def clean_text(text):
#     text = re.sub(r"http\S+", "", text)  # remove URLs
#     text = re.sub(r"\s+", " ", text)  # remove extra spaces
#     text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove non-ASCII
#     return text.strip()


# df["text"] = df["text"].apply(clean_text)

# # Keep only needed columns for training
# df_clean = df[["label", "text"]]

# # Save cleaned dataset
# df_clean.to_csv(SAVE_PATH, index=False)
# print(f"Cleaned dataset saved at {SAVE_PATH}")
