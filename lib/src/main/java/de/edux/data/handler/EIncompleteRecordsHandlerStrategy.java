package de.edux.data.handler;

public enum EIncompleteRecordsHandlerStrategy {
  DO_NOT_HANDLE(new DoNotHandleIncompleteRecords()),
  DROP_RECORDS(new DropIncompleteRecordsHandler()),
  FILL_RECORDS_WITH_AVERAGE(new AverageFillIncompleteRecordsHandler());

  private final IIncompleteRecordsHandler incompleteRecordHandler;

  EIncompleteRecordsHandlerStrategy(IIncompleteRecordsHandler incompleteRecordHandler) {
    this.incompleteRecordHandler = incompleteRecordHandler;
  }

  public IIncompleteRecordsHandler getHandler() {
    return this.incompleteRecordHandler;
  }
}
