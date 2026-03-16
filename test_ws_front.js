const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => {
    for (let i = 0; i < msg.args().length; ++i)
      console.log(`${i}: ${msg.args()[i]}`);
  });
  
  page.on('pageerror', error => {
    console.log(`pageerror: ${error.message}`);
  });
  
  page.on('requestfailed', request => {
    console.log(`requestfailed: ${request.url()} - ${request.failure().errorText}`);
  });

  await page.goto('http://localhost:8000');
  
  // Click on recognize tab
  await page.evaluate(() => loadPage('recognize'));
  
  // Wait 10 seconds for reconnects to happen
  await new Promise(r => setTimeout(r, 10000));
  
  await browser.close();
})();
