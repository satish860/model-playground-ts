import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { generateText } from 'ai';
import * as dotenv from 'dotenv';

dotenv.config();

async function main() {
  const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY || 'YOUR_OPENROUTER_API_KEY',
  });

  const { text } = await generateText({
    model: openrouter.chat('minimax/minimax-m2:free'),
    prompt: 'What is OpenRouter?',
  });

  console.log(text);
}

main().catch(console.error);
