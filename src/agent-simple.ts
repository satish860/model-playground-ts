import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { generateText, tool } from 'ai';
import { z } from 'zod';
import * as dotenv from 'dotenv';

dotenv.config();

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || 'YOUR_OPENROUTER_API_KEY',
});

// Simple tool with minimal parameters
const simpleTools = {
  getTime: tool({
    description: 'Get the current time',
    parameters: z.object({}),
    execute: async () => {
      const time = new Date().toLocaleTimeString();
      console.log(`[TOOL EXECUTED] getTime() returned: ${time}`);
      return `Current time is ${time}`;
    },
  }),

  add: tool({
    description: 'Add two numbers together',
    parameters: z.object({
      a: z.number().describe('First number'),
      b: z.number().describe('Second number'),
    }),
    execute: async (params) => {
      console.log(`[TOOL EXECUTED] add() received:`, params);
      const { a, b } = params;
      const result = a + b;
      console.log(`[TOOL EXECUTED] add(${a}, ${b}) = ${result}`);
      return `${a} + ${b} = ${result}`;
    },
  }),
};

async function testSimpleAgent() {
  console.log('Testing simple agent with MiniMax M2...\n');

  try {
    const result = await generateText({
      model: openrouter.chat('minimax/minimax-m2'),
      prompt: 'What is 5 + 3? Use the add tool to calculate it, then tell me the answer.',
      tools: simpleTools,
      maxSteps: 5,
      onStepFinish: (step) => {
        console.log('\n--- STEP FINISHED ---');
        console.log('FULL STEP OBJECT:');
        console.log(JSON.stringify(step, null, 2));
        console.log('\n--- PARSED DATA ---');
        console.log('Step number:', step.stepNumber || 'unknown');
        console.log('Text:', step.text || '(none)');
        console.log('Tool calls:', step.toolCalls?.length || 0);
        if (step.toolCalls) {
          step.toolCalls.forEach((call, i) => {
            console.log(`  Tool ${i + 1}:`, call.toolName);
            console.log(`  Full call object:`, JSON.stringify(call, null, 2));
          });
        }
        console.log('Tool results:', step.toolResults?.length || 0);
        if (step.toolResults) {
          step.toolResults.forEach((result, i) => {
            console.log(`  Result ${i + 1}:`, JSON.stringify(result, null, 2));
          });
        }
        console.log('Finish reason:', step.finishReason);
        console.log('---\n');
      },
    });

    console.log('\n=== FINAL RESULT ===');
    console.log('FULL RESULT OBJECT:');
    console.log(JSON.stringify(result, null, 2));
    console.log('\n--- PARSED RESULT ---');
    console.log('Text:', result.text || '(none)');
    console.log('Steps:', result.steps?.length || 0);
    console.log('Finish reason:', result.finishReason);
    console.log('Total tokens:', result.usage?.totalTokens || 0);

  } catch (error) {
    console.error('ERROR:', error);
    if (error instanceof Error) {
      console.error('Stack:', error.stack);
    }
  }
}

testSimpleAgent();
