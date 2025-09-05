import asyncio
import os
import shutil
from playwright.async_api import async_playwright
import kagglehub

EMAIL = "your_email_here"
PASSWORD = "your_password_here"
QUERY = "basketball dataset"
DOWNLOAD_PATH = "downloads"

async def get_dataset_slug(playwright):
    """Automate Kaggle search and return first dataset slug"""
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()

    # Go to Kaggle homepage
    await page.goto("https://www.kaggle.com/")
    print("🌐 Opened Kaggle homepage")

    # Click Sign In
    await page.click("text=Sign In")
    print("🔐 Clicked Sign In")

    # Click "Sign in with Email"
    await page.click("text=Sign in with Email")

    # Fill credentials
    await page.fill("input[name='email']", EMAIL)
    await page.fill("input[name='password']", PASSWORD)

    # Submit login
    await page.click("button:has-text('Sign In')")
    print("🔑 Submitted login form")

    # Wait until login succeeds
    await page.wait_for_selector("a[href*='/account'], img[alt*='avatar']")
    print("✅ Logged in successfully")

    # Go to datasets search
    await page.goto("https://www.kaggle.com/datasets")
    await page.fill("input[placeholder='Search datasets']", QUERY)
    await page.keyboard.press("Enter")
    await page.wait_for_selector("a[href*='/datasets/']")

    # Get the first dataset link
    dataset_url = await page.get_attribute("a[href*='/datasets/'] >> nth=0", "href")
    slug = dataset_url.replace("/datasets/", "").strip()
    print(f"📂 Dataset page: https://www.kaggle.com{dataset_url}")
    print(f"🔗 Extracted dataset slug: {slug}")

    await browser.close()
    return slug

def clear_cache():
    """Clear kagglehub cache directory"""
    cache_dir = os.path.expanduser("~/.cache/kagglehub")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("🧹 Cleared KaggleHub cache directory")

def download_dataset(slug):
    """Download dataset with Kaggle API, move it, then clean up"""
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    # Download with kagglehub
    path = kagglehub.dataset_download(slug)
    print(f"📥 Downloaded to cache: {path}")

    # Copy files to downloads folder
    for filename in os.listdir(path):
        src = os.path.join(path, filename)
        dst = os.path.join(DOWNLOAD_PATH, filename)
        shutil.copy2(src, dst)

    print(f"✅ Dataset saved in: {os.path.abspath(DOWNLOAD_PATH)}")

    # After copying, delete the temp dataset folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("🗑️ Cleared temporary dataset folder")

    # Also clear cache
    clear_cache()

async def main():
    async with async_playwright() as playwright:
        slug = await get_dataset_slug(playwright)
        download_dataset(slug)

if __name__ == "__main__":
    asyncio.run(main())
