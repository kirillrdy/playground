#import "bridge.m"
#include <assert.h>

size_t sam_query_completions(const char *text, size_t caret, size_t *start,
                            size_t *end, const char **output, size_t capacity) {
    return 0;
}

// Reproduce AppKit requesting a cell synchronously while a column is added.
@interface ReentrantResultsTable : NSTableView
@end
@implementation ReentrantResultsTable
- (void)addTableColumn:(NSTableColumn *)column {
    assert(g_delegate.resultRows.count == 0);
    NSTextField *cell = (NSTextField *)[g_delegate tableView:self viewForTableColumn:column row:0];
    assert([cell.stringValue isEqualToString:@""]);
    [super addTableColumn:column];
}
@end

static void updateTable(const char *json) {
    __block BOOL finished = NO;
    sam_macos_set_query_table(json);
    dispatch_async(dispatch_get_main_queue(), ^{ finished = YES; });
    while (!finished) {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
    }
}

static NSString *cellValue(NSInteger row, NSUInteger column) {
    NSTextField *cell = (NSTextField *)[g_delegate tableView:g_delegate.resultsTable
        viewForTableColumn:g_delegate.resultsTable.tableColumns[column] row:row];
    return cell.stringValue;
}

int main(void) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        g_delegate = [[SamAppDelegate alloc] initWithCallbacks:NULL];
        g_delegate.resultsTable = [[ReentrantResultsTable alloc] init];
        g_delegate.resultsTable.dataSource = g_delegate;
        g_delegate.resultsTable.delegate = g_delegate;
        g_delegate.resultsScroll = [[NSScrollView alloc] init];
        g_delegate.canvasView = [[SamCanvasView alloc] init];

        updateTable("{\"columns\":[\"timestamp\"],\"rows\":[[\"0.033000\"]]}");
        assert([cellValue(0, 0) isEqualToString:@"0.033000"]);
        updateTable("{\"columns\":[\"timestamp\",\"frame_id\"],\"rows\":[]}");
        assert(g_delegate.resultsTable.tableColumns.count == 2);
        updateTable("{\"columns\":[\"timestamp\",\"frame_id\"],\"rows\":[[\"0.066000\",\"2\"]]}");
        assert([cellValue(0, 0) isEqualToString:@"0.066000"]);
        assert([cellValue(0, 1) isEqualToString:@"2"]);
        updateTable("{\"columns\":[\"timestamp\"],\"rows\":[[\"0.099000\"]]}");
        assert(g_delegate.resultsTable.tableColumns.count == 1);
        assert([cellValue(0, 0) isEqualToString:@"0.099000"]);
        updateTable("{\"columns\":[\"timestamp\"],\"rows\":[]}");
        assert([cellValue(0, 0) isEqualToString:@""]);
        assert([cellValue(-1, 0) isEqualToString:@""]);
        updateTable("null");
        assert(g_delegate.resultsScroll.hidden);
        assert(!g_delegate.canvasView.hidden);
        NSLog(@"Query table schema transition tests passed");
    }
    return 0;
}
