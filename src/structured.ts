import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { generateObject } from 'ai';
import { z } from 'zod';
import * as dotenv from 'dotenv';

dotenv.config();

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || 'YOUR_OPENROUTER_API_KEY',
});

async function example1_BasicStructuredData() {
  console.log('\n=== Example 1: Basic Structured Data ===\n');

  const { object } = await generateObject({
    model: openrouter.chat('minimax/minimax-m2'),
    schema: z.object({
      recipe: z.object({
        name: z.string(),
        ingredients: z.array(
          z.object({
            name: z.string(),
            amount: z.string(),
          }),
        ),
        steps: z.array(z.string()),
      }),
    }),
    prompt: 'Generate a recipe for chocolate chip cookies. Output as JSON.',
  });

  console.log('Recipe:', JSON.stringify(object, null, 2));
}

async function example2_ComplexStructuredData() {
  console.log('\n=== Example 2: Complex Structured Data ===\n');

  const { object } = await generateObject({
    model: openrouter.chat('minimax/minimax-m2'),
    schema: z.object({
      characters: z.array(
        z.object({
          name: z.string(),
          class: z.enum(['warrior', 'mage', 'archer', 'rogue']),
          level: z.number(),
          stats: z.object({
            strength: z.number(),
            intelligence: z.number(),
            dexterity: z.number(),
          }),
          equipment: z.array(z.string()),
        }),
      ),
    }),
    prompt: 'Generate 3 RPG characters with different classes. Output as JSON.',
  });

  console.log('Characters:', JSON.stringify(object, null, 2));
}

async function example3_NotificationData() {
  console.log('\n=== Example 3: Notification Data ===\n');

  const { object } = await generateObject({
    model: openrouter.chat('minimax/minimax-m2'),
    schema: z.object({
      notification: z.object({
        title: z.string(),
        body: z.string(),
        actions: z.array(
          z.object({
            label: z.string(),
            action: z.string(),
          }),
        ),
      }),
    }),
    prompt: 'Generate a notification for a user receiving a new message. Output as JSON.',
  });

  console.log('Notification:', JSON.stringify(object, null, 2));
}

async function example4_ArrayData() {
  console.log('\n=== Example 4: Array of Items ===\n');

  const { object } = await generateObject({
    model: openrouter.chat('minimax/minimax-m2'),
    schema: z.object({
      tasks: z.array(
        z.object({
          id: z.number(),
          title: z.string(),
          priority: z.enum(['low', 'medium', 'high']),
          estimatedHours: z.number(),
          tags: z.array(z.string()),
        }),
      ),
    }),
    prompt: 'Generate 5 software development tasks for building a todo app. Output as JSON.',
  });

  console.log('Tasks:', JSON.stringify(object, null, 2));
}

async function main() {
  try {
    await example1_BasicStructuredData();
    await example2_ComplexStructuredData();
    await example3_NotificationData();
    await example4_ArrayData();
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : String(error));
  }
}

main();
