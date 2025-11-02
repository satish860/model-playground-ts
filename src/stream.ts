import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { streamText } from 'ai';
import * as dotenv from 'dotenv';

dotenv.config();

async function main() {
  const openrouter = createOpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY || 'YOUR_OPENROUTER_API_KEY',
  });

  const result = streamText({
    model: openrouter.chat('minimax/minimax-m2:free'),
    prompt: 'Write a short story about AI.',
  });

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  console.log('\n\nStreaming complete!');
}

main().catch(console.error);
