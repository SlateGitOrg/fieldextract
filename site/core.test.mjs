import test from 'node:test';
import assert from 'node:assert/strict';
import { compute, initialState, alternateState } from './core.mjs';

test('initial scenario returns the expected decision contract',async()=>{const value=await compute(structuredClone(initialState));assert.equal(value.status,"Review required");assert.equal(typeof value.summary,'string');assert.ok(value.metrics.length>=2);assert.ok(value.rows.length>=1)});

test('changed input produces changed output',async()=>{const before=await compute(structuredClone(initialState));const after=await compute(structuredClone(alternateState));assert.notEqual(JSON.stringify(before),JSON.stringify(after))});
