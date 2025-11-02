import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { generateText, tool } from 'ai';
import { z } from 'zod';
import * as dotenv from 'dotenv';

dotenv.config();

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || 'YOUR_OPENROUTER_API_KEY',
});

// Define tools for the agent
const tools = {
  calculator: tool({
    description: 'Perform basic arithmetic operations (add, subtract, multiply, divide)',
    parameters: z.object({
      operation: z.enum(['add', 'subtract', 'multiply', 'divide']).describe('The arithmetic operation to perform'),
      num1: z.number().describe('First number'),
      num2: z.number().describe('Second number'),
    }),
    execute: async (args) => {
      console.log(`  [Tool] calculator received args:`, JSON.stringify(args));
      const { operation, num1, num2 } = args;
      let result: number;
      switch (operation) {
        case 'add':
          result = num1 + num2;
          break;
        case 'subtract':
          result = num1 - num2;
          break;
        case 'multiply':
          result = num1 * num2;
          break;
        case 'divide':
          if (num2 === 0) throw new Error('Cannot divide by zero');
          result = num1 / num2;
          break;
        default:
          throw new Error(`Unknown operation: ${operation}`);
      }
      console.log(`  [Tool] calculator(${operation}, ${num1}, ${num2}) = ${result}`);
      return `The result of ${num1} ${operation} ${num2} is ${result}`;
    },
  }),

  getWeather: tool({
    description: 'Get the current weather for a city',
    parameters: z.object({
      city: z.string().describe('The city name'),
    }),
    execute: async ({ city }) => {
      // Simulate weather API call
      const weather = {
        temperature: Math.floor(Math.random() * 30) + 10,
        condition: ['sunny', 'cloudy', 'rainy', 'snowy'][Math.floor(Math.random() * 4)],
        humidity: Math.floor(Math.random() * 40) + 40,
      };
      console.log(`  [Tool] getWeather("${city}") = ${JSON.stringify(weather)}`);
      return `The weather in ${city} is ${weather.condition} with a temperature of ${weather.temperature}°C and ${weather.humidity}% humidity.`;
    },
  }),

  searchKnowledge: tool({
    description: 'Search a knowledge base for information',
    parameters: z.object({
      query: z.string().describe('The search query'),
    }),
    execute: async ({ query }) => {
      // Simulate knowledge base search
      const knowledge = {
        'typescript': 'TypeScript is a strongly typed programming language that builds on JavaScript.',
        'ai': 'Artificial Intelligence is the simulation of human intelligence by machines.',
        'nodejs': 'Node.js is a JavaScript runtime built on Chrome\'s V8 JavaScript engine.',
      };

      const lowerQuery = query.toLowerCase();
      for (const [key, value] of Object.entries(knowledge)) {
        if (lowerQuery.includes(key)) {
          console.log(`  [Tool] searchKnowledge("${query}") = Found match for "${key}"`);
          return value;
        }
      }

      console.log(`  [Tool] searchKnowledge("${query}") = No results found`);
      return 'No relevant information found in the knowledge base.';
    },
  }),

  getCurrentTime: tool({
    description: 'Get the current date and time',
    parameters: z.object({}),
    execute: async () => {
      const now = new Date();
      console.log(`  [Tool] getCurrentTime() = ${now.toLocaleString()}`);
      return now.toLocaleString();
    },
  }),
};

async function runAgent(userPrompt: string, maxSteps: number = 5) {
  console.log('='.repeat(80));
  console.log('AGENT EXECUTION');
  console.log('='.repeat(80));
  console.log(`User Prompt: ${userPrompt}`);
  console.log(`Max Steps: ${maxSteps}`);
  console.log('='.repeat(80));
  console.log('');

  const startTime = Date.now();

  try {
    const result = await generateText({
      model: openrouter.chat('minimax/minimax-m2'),
      prompt: userPrompt,
      tools: tools,
      maxSteps: maxSteps,
      onStepFinish: ({ text, toolCalls, toolResults, finishReason, usage }) => {
        console.log('\n--- Step Completed ---');
        if (text) {
          console.log(`Text: ${text.substring(0, 200)}${text.length > 200 ? '...' : ''}`);
        }
        if (toolCalls && toolCalls.length > 0) {
          console.log(`Tool Calls: ${toolCalls.length}`);
          toolCalls.forEach((call, index) => {
            console.log(`  ${index + 1}. ${call.toolName}`);
            if (call.args) {
              console.log(`     Args: ${JSON.stringify(call.args, null, 2)}`);
            } else {
              console.log(`     Args: (none)`);
            }
          });
        }
        if (toolResults && toolResults.length > 0) {
          console.log(`Tool Results: ${toolResults.length}`);
          toolResults.forEach((result, index) => {
            const resultStr = typeof result.result === 'string'
              ? result.result
              : JSON.stringify(result.result);
            const displayStr = resultStr ? resultStr.substring(0, 100) : '(empty)';
            console.log(`  ${index + 1}. ${result.toolName}: ${displayStr}`);
          });
        }
        console.log(`Finish Reason: ${finishReason || 'unknown'}`);
        console.log(`Tokens Used: ${usage?.totalTokens || 0}`);
        console.log('');
      },
    });

    const duration = Date.now() - startTime;

    console.log('='.repeat(80));
    console.log('FINAL RESULT');
    console.log('='.repeat(80));
    console.log(`\n${result.text || '(no text response)'}\n`);
    console.log('='.repeat(80));
    console.log(`Total Steps: ${result.steps?.length || 0}`);
    console.log(`Total Tokens: ${result.usage?.totalTokens || 0}`);
    console.log(`Duration: ${duration}ms`);
    console.log(`Finish Reason: ${result.finishReason || 'unknown'}`);
    console.log('='.repeat(80));
  } catch (error) {
    console.error('\n=== ERROR ===');
    console.error('Error during agent execution:');
    console.error(error);
    if (error instanceof Error) {
      console.error('Stack trace:', error.stack);
    }
    console.error('='.repeat(80));
  }
}

async function main() {
  try {
    // Example 1: Multi-step calculation
    await runAgent(
      'Calculate (15 + 25) * 3 and then divide the result by 10. Show me each step.',
      10
    );

    console.log('\n\n');

    // Example 2: Information retrieval and reasoning
    await runAgent(
      'What is TypeScript? After you find out, tell me the current time.',
      5
    );

    console.log('\n\n');

    // Example 3: Complex multi-tool task
    await runAgent(
      'Get the weather in London, then calculate what the temperature would be in Fahrenheit (use formula: F = C * 9/5 + 32).',
      10
    );

  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : String(error));
  }
}

main();
