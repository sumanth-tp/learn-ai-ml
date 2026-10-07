import { expect, test } from 'claude-code/testing'

test('denies an edit to a .env file', async $ => {
  const ran = await $.tool.call({
    tool: 'Edit',
    file_path: '/project/.env',
    old_string: 'A=1',
    new_string: 'A=2',
  })

  expect(ran.deny).toContain('protected')
})
