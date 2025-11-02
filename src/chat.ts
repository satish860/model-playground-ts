import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { streamText, CoreMessage } from 'ai';
import * as dotenv from 'dotenv';
import * as readline from 'readline';

dotenv.config();

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || 'YOUR_OPENROUTER_API_KEY',
});

// Store conversation history
const messages: CoreMessage[] = [];

// Create readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function chat(userMessage: string) {
  // Add user message to history
  messages.push({
    role: 'user',
    content: userMessage,
  });

  console.log('\nAssistant: ');

  // Stream the response
  const { textStream } = await streamText({
    model: openrouter.chat('minimax/minimax-m2:free'),
    messages: messages,
  });

  let fullResponse = '';

  for await (const chunk of textStream) {
    process.stdout.write(chunk);
    fullResponse += chunk;
  }

  console.log('\n');

  // Add assistant response to history
  messages.push({
    role: 'assistant',
    content: fullResponse,
  });
}

async function main() {
  console.log('='.repeat(60));
  console.log('MiniMax M2 Command Line Chat');
  console.log('='.repeat(60));
  console.log('Type your message and press Enter to chat.');
  console.log('Type "exit" or "quit" to end the conversation.');
  console.log('Type "clear" to clear conversation history.');
  console.log('='.repeat(60));
  console.log('');

  const askQuestion = () => {
    rl.question('You: ', async (input) => {
      const trimmedInput = input.trim();

      if (trimmedInput.toLowerCase() === 'exit' || trimmedInput.toLowerCase() === 'quit') {
        console.log('\nGoodbye!');
        rl.close();
        return;
      }

      if (trimmedInput.toLowerCase() === 'clear') {
        messages.length = 0;
        console.log('\nConversation history cleared.\n');
        askQuestion();
        return;
      }

      if (trimmedInput === '') {
        askQuestion();
        return;
      }

      try {
        await chat(trimmedInput);
        askQuestion();
      } catch (error) {
        console.error('Error:', error instanceof Error ? error.message : String(error));
        askQuestion();
      }
    });
  };

  askQuestion();
}

main();
